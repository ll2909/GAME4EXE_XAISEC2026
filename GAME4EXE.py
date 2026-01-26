import torch
import numpy as np
import random
from typing import Union, Optional
import time
import pefile
import explainers.SimpleGradients as grad
from preprocessing.FilePreprocessor import load_and_preprocess_file


device = "cuda" if torch.cuda.is_available() else "cpu"


def set_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reconstruct_perturbation_bytes(model, emb_adv):
    emb_values = torch.tensor([i for i in range(256)], dtype=torch.int32)
    emb_values = model.embedding_1(emb_values.to(device)).detach()
    p_bytes = []
    for e in emb_adv:
        # find the index of the most similar embedding between e and emb_values
        idx = torch.argmin(torch.norm(e - emb_values, dim=1)).cpu().item()
        p_bytes.append(idx)

    return bytes(p_bytes)


class DOSHeaderXAIEvasion():
    def __init__(self, model, softplus_beta : float = 10.0, ):
        self.model = model.to(device).eval()
        self.softplus_beta = softplus_beta
        # Replace ReLU with Softplus
        print("Replacing ReLU with Softplus")
        for idx, c in enumerate(model.named_children()):
            if type(c[1]) == torch.nn.modules.activation.ReLU:
                setattr(self.model, c[0], torch.nn.Softplus(beta = self.softplus_beta))
        print("----- MODIFIED MODEL FOR ATTACK -----")
        print(self.model)


    def generate_adversarial(self, 
                             malware_path : str, 
                             goodware_path : Optional[str], 
                             output_path : str, 
                             target_label : Union[float,int] = 0, 
                             target_expl : torch.Tensor = None, 
                             n_steps : int = 100,
                             input_size = 2**20, 
                             lr : float = 1e-2, 
                             lambda_p = 1.0, 
                             lambda_x = 1.0,
                             seed = 42, 
                             patience = 5, 
                             verbose = False
                             ):
        

        set_reproducibility(seed)
        slice_dos_header = slice(2, 60)

        orig_x, f_bytez = load_and_preprocess_file(malware_path, max_dim=input_size, pad_value=0)
        pe = pefile.PE(data = f_bytez)
        # Get PE Header offset
        pe_offset = pe.NT_HEADERS.get_file_offset()
        pe.close()
        # Find the DOS Stub section
        slice_dos_stub = slice(64, pe_offset)
        max_changable_bytes = 60 - 2 + pe_offset-64
        print("Max changable bytes: ", max_changable_bytes)


        # Predict the original malware
        emb_orig_x = self.model.embed(orig_x.to(device)).detach()
        m_conf = self.model(emb_orig_x).squeeze().detach().cpu().item()
        m_pred = int(0 if m_conf <= 0.5 else 1)
        print(f"Malware Confidence: {m_conf}, Prediction: {m_pred}")


        target_g, g_bytez = load_and_preprocess_file(goodware_path, max_dim=input_size, pad_value=0)
        emb_target_g = self.model.embed(target_g.to(device)).detach()
        g_conf = self.model(emb_target_g).squeeze().cpu().item()
        g_pred = int(0 if g_conf <= 0.5 else 1)
        print(f"Target Goodware Confidence: {g_conf}, Prediction: {g_pred}")
        if g_pred != 0:
            raise Exception("Target Goodware predicted as malware.")



        # Perturbations initialization: the best strategy is to initialize using target goodware - original malware header bytes 
        deltas = [torch.nn.Parameter((emb_target_g[0, slice_dos_header,:] - emb_orig_x[0, slice_dos_header,:]), requires_grad=True)]
        if max_changable_bytes > 58:
            deltas.append(torch.nn.Parameter((emb_target_g[0, slice_dos_stub,:] - emb_orig_x[0,slice_dos_stub,:]), requires_grad=True))                
        else:
            print("Only the DOS header is changable, ignoring DOS stub manipulation.")


        y_loss = torch.nn.MSELoss()
        xai_loss = torch.nn.MSELoss(reduction="sum")
        
        optim = torch.optim.Adam(deltas, lr = lr)

        adv_x = orig_x.clone().to(device)

        if target_label == "same":
            print("The attack WILL NOT alter prediction.")
            target_y = torch.tensor(m_conf, dtype=torch.float, device = device)
        else:
            target_y  = torch.tensor(g_conf, dtype=torch.float, device = device)

        explainer = grad.SimpleGradients(self.model, multiply=False)
        
        if target_expl is not None:
            expl_t = target_expl.to(device)
        else:
            emb_t = self.model.embed(target_g.to(device))
            expl_t = explainer.attribute(emb_t)    
        
        best_loss = torch.inf
        start_time = time.time()
        early_stopping = 0

        tot_losses = []
        p_losses = []
        ig_losses = []
        

        expl_t = expl_t.flatten().detach()

        torch.cuda.empty_cache()

        for i in range(n_steps):
            optim.zero_grad()

            emb_adv = self.model.embed(adv_x).detach()

            emb_adv[0,slice_dos_header,:] = emb_adv[0,slice_dos_header,:] + deltas[0]   # Apply pertubation by addition
            if len(deltas) > 1:
                emb_adv[0,slice_dos_stub,:] = emb_adv[0,slice_dos_stub,:] + deltas[1]   # Apply pertubation by addition

            conf = self.model(emb_adv, is_embedded = True).squeeze()
            
            if lambda_p != 0.0:
                p_loss = y_loss(conf, target_y) * lambda_p
            else:
                conf = conf.detach().cpu()
                p_loss = torch.tensor(0.0)
            
            if lambda_x != 0.0:
                expl_adv = explainer.attribute(emb_adv)
                expl_adv = expl_adv.flatten()
                x_loss = xai_loss(expl_adv, expl_t) * lambda_x
            else:
                x_loss = torch.tensor(0.0)

            total_loss = p_loss + x_loss

            tot_losses.append(total_loss.item())
            p_losses.append(p_loss.item())
            ig_losses.append(x_loss.item())

            
            print("Step %i\n Confidence: %f\t Total Loss: %1.16f\t Pred. Loss: %1.16f\t Expl. Loss: %1.16f" % (i+1, conf, total_loss, p_loss.item(), x_loss.item()))
            

            if total_loss.item() < best_loss:
                best_step = i+1
                best_loss = total_loss
                best_x_loss = x_loss.item()
                best_p_loss = p_loss.item()
                best_deltas = deltas.copy()
                early_stopping = 0
            else:
                early_stopping = early_stopping + 1
            if early_stopping > patience-1:
                print("Stagnating loss, early stopping triggered at step %i." % (i+1))
                break
            
            
            total_loss.backward()

            optim.step()

            torch.cuda.empty_cache()
        
        if verbose:
            print("Best Delta found at step %i: %f" % (best_step, best_loss.item()))

        orig_emb = emb_orig_x

        if len (best_deltas) > 1:
            delta_1_bytes = reconstruct_perturbation_bytes(self.model, orig_emb[0,slice_dos_header,:] + best_deltas[0])
            delta_2_bytes = reconstruct_perturbation_bytes(self.model, orig_emb[0,slice_dos_stub,:] + best_deltas[1])
            adv_bytez = f_bytez[:2] + delta_1_bytes + f_bytez[60:64] + delta_2_bytes + f_bytez[pe_offset:]
            diff_1_bytes = sum([x != y for x, y in zip(f_bytez[slice_dos_header], delta_1_bytes)])
            diff_2_bytes = sum([x != y for x, y in zip(f_bytez[slice_dos_stub], delta_2_bytes)])
            diff_bytes = diff_1_bytes + diff_2_bytes
        else:
            delta_1_bytes = reconstruct_perturbation_bytes(self.model, orig_emb[0,slice_dos_header,:] + best_deltas[0])
            adv_bytez = f_bytez[:2] + delta_1_bytes + f_bytez[60:]
            diff_bytes = sum([x != y for x, y in zip(f_bytez[slice_dos_header], delta_1_bytes)])
        
        
        print("Saving adversarial example")
        with open(output_path, "wb") as adv_f:
            adv_f.write(adv_bytez)

        print("Manipulation done in %f seconds\n" % (time.time() - start_time))
        torch.cuda.empty_cache()
        print("Changed %i bytes" % diff_bytes)

        report = {
            "filename" : malware_path.split("/")[-1],
            "best_step": best_step,
            "total_loss": best_loss.item(),
            "pred_loss": best_p_loss,
            "expl_loss": best_x_loss,
            "changed_bytes": diff_bytes,
            "max_changable_bytes": max_changable_bytes,
        }

        return report
