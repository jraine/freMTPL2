import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np
import pathlib

def get_activation(name, *args, **kwargs):
    actdict = {
        "linear": lambda x: x,
        "relu": F.relu,
        "leaky_relu": F.leaky_relu,
        "sigmoid": F.sigmoid,
        "selu": F.selu,
        "celu": F.celu,
        "elu": F.elu,
        "swish": F.hardswish,
        "softplus": F.softplus,
    }
    assert name.lower() in actdict, f"Currently {name} is not supported.  Choose one of '{actdict.keys()}'"

    return actdict[name.lower()]

class DenseNet(nn.Module):

    def __init__(self, in_dim, out_dim, n_cond=0,
                 nodes_per_layer=32, num_layers=2, nodelist=None,  
                 hidden_act='relu', output_act='linear',
                 batch_norm=False, layer_norm=False):
        super().__init__()

        if nodelist in  ['None',None]:
            nodelist = [nodes_per_layer]*num_layers
        
        if n_cond > 0:
            in_dim += n_cond
        
        ins = [in_dim] + nodelist
        outs = nodelist + [out_dim]
        self.layers = nn.ModuleList([nn.Linear(i,o) for i,o in zip(ins,outs)])
        self.acts   = [get_activation(hidden_act) for i in ins] + [get_activation(output_act)]
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

    def forward(self, x, cond=None):

        if cond is not None:
            x = torch.cat([x,cond],axis=1)

        for l,a in zip(self.layers[:-1],self.acts[:-1]):
            x = a(l(x))
            if self.batch_norm:
                x = F.batch_norm(x)
            if self.layer_norm:
                x = F.layer_norm(x)
        x = self.acts[-1](self.layers[-1](x))
        return x



def get_dense(inp_dim, out_dim, n_cond=0,
              nodes=32, layers=2, nodelist=None,
              hidden_act='relu', output_act='linear',
              batch_norm=False, layer_norm=False):

    return DenseNet(inp_dim, out_dim, n_cond,
                    nodes, layers, nodelist,
                    hidden_act, output_act,
                    batch_norm, layer_norm)



def train_class(model, train_data, val_data, n_epochs, learning_rate, 
                cond=False, class_weights=None,
                path=None, name=None, checkpoint=False,
                multiclass=False, loss_fig=True, device='cpu'):
    path = pathlib.Path(path)
    save_path = pathlib.Path(path / name)
    save_path.mkdir(parents=True, exist_ok=True)
    if checkpoint:
        chkp_path = pathlib.Path(path / name / 'checkpoints/')
        chkp_path.mkdir(parents=True, exist_ok=True)


    model.to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = np.zeros(n_epochs)
    val_loss = np.zeros(n_epochs)

    loss_fn = torch.nn.CrossEntropyLoss

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        ep_loss = []
        for step, data in enumerate(train_data):
            model.train()
            opt.zero_grad()
            if cond:
                x,c,y = [d.to(device) for d in data]
            else:
                x,y = [d.to(device) for d in data]
                c = None
            yp = model(x,cond=c)

            loss = loss_fn(yp.squeeze(),y.squeeze(),class_weights=[2-class_weights,class_weights])
            loss.backward()
            opt.step()
            ep_loss.append(loss.item())
        train_loss[epoch] = np.array(ep_loss).mean()
        
        vloss = []
        for step, data in enumerate(val_data):
            with torch.no_grad():
                x,y = [d.to(device) for d in data]
                yp = model(x)
                vloss.append(loss_fn(yp.squeeze(),y.squeeze()).item())
        val_loss[epoch] = np.array(vloss).mean()

        print(f"loss: {train_loss[epoch]:.3f}\t val_loss: {val_loss[epoch]:.3f}")
        if checkpoint:
            torch.save(model.state_dict(), chkp_path / f'epoch_{epoch}_valloss_{val_loss[epoch]:.3f}.pt')

    torch.save(model.state_dict(), save_path / f'model_valloss_{val_loss[epoch]:.3f}.pt')
    # print(f"Loss = {train_loss[epoch]:.3f},\t val_loss = {val_loss[epoch]:.3f}")

    return train_loss, val_loss