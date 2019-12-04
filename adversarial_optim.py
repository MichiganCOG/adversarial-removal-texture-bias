import torch
import torch.optim as optim
import copy
import numpy as np
from contextlib import contextmanager


######################################################
#               ADVERSARIAL OPTIMIZER                #
######################################################

class AdversarialWrapper(optim.Optimizer):
    def __init__(self, task_optim, adversary_optim, eta=1):
        if eta < 1:
            raise ValueError("Invalid eta: {}".format(eta))
        
        params = task_optim.param_groups + adversary_optim.param_groups
    
        # Adds task and adversary parameters to self.param_groups and ensures they don't overlap
        #  - self.param_groups is a list of dicts, wherein params are stored under the key 'params'
        #  - Example: param_list = [p for g in self.param_groups for p in g['params']]
        super(AdversarialWrapper, self).__init__(params, defaults=dict())
        
        # Keep copies of each parameter, and the last step taken
        self._copy_params = copy.deepcopy(self.param_groups)
        self._copy_params = [p for g in self._copy_params for p in g['params']] # Listify
        self._last_diff = copy.deepcopy(self.param_groups)
        self._last_diff = [p for g in self._last_diff for p in g['params']]
        self.reset() # Zero out last_diff
        
        # Task and adversary optimizers
        self._task_optim = task_optim
        self._adv_optim = adversary_optim
        # Number of adversary steps per task step
        self._eta = eta
        # Internal step count since last task step
        self._steps_since_task = self._eta-1
            
    # Update our internal copies of the parameters. Must be called after every parameter 
    #   change if gradient prediction is being used; can be ignored otherwise
    def update(self):
        for i, param in enumerate([p for g in self.param_groups for p in g['params']]):
            self._last_diff[i].data[:] = param.data[:] - self._copy_params[i].data[:]
            self._copy_params[i].data[:] = param.data[:]
    
    # Reset our internal copies of the parameters. Useful when switching between pretraining and
    #   training, or when turning prediction on/off
    def reset(self):
        for i, param in enumerate([p for g in self.param_groups for p in g['params']]):
            self._copy_params[i].data[:] = param.data[:]
            self._last_diff[i].data[:] = 0.0 
        
    # Update the task parameters (featurizer and classifier) by calling the task optimizer's step()
    def step_task(self, update_after=True, **kwargs):
        self._task_optim.step(**kwargs)
        self._steps_since_task = 0
        if update_after:
            self.update()
    
    # Update the adversary parameters (discriminator only) by calling the adversary optimizer's step()
    def step_adversary(self, update_after=True, **kwargs):
        self._adv_optim.step(**kwargs)
        self._steps_since_task += 1
        if update_after:
            self.update()
    
    # The Optimizer class method. Alternates between 1 task step and [eta] adversary steps
    def step(self, update_after=True, **kwargs):
        if self._steps_since_task >= self._eta:
            self.step_task(update_after, **kwargs)
        else:
            self.step_adversary(update_after, **kwargs)
        
    # Look ahead in parameter space to compute gradients at a predicted point
    @contextmanager
    def lookahead(self, step=1.0):
        # If step is 0.0, do nothing
        if step == 0.0:
            yield
            return
        
        # Otherwise, step each parameter forward
        param_list = [p for g in self.param_groups for p in g['params']]
        for i, p in enumerate(param_list):
            # Integrity check
            if torch.sum(p.data[:] != self._copy_params[i].data[:]) > 0:
                raise RuntimeWarning("Stored parameters differ from current ones. Use step(update=True) when taking an optimization step, or manually call update() after each modification to the network parameters.")
            
            p.data[:] += step * self._last_diff[i].data[:]
        
        yield
        
        # Roll back to original values
        for i, p in enumerate(param_list):
            p.data[:] = self._copy_params[i].data[i]
            

