import os 
from torch import nn
import torch
from torch.optim.optimizer import Optimizer

def load_saved_weights(model, model_path, load_classifier=True):
    # The current model and the model to initialize from can have different number of classes.
    state_dict = torch.load(model_path)
    del state_dict['classifier.weight']
    model.load_state_dict(state_dict, strict=False)

    if load_classifier:
        state_dict = torch.load(model_path)
        model.classifier.weight.data[:len(state_dict['classifier.weight'])] = state_dict['classifier.weight']

def save_checkpoint(output_dir, model, name='base_model_epoch', epoch=None):
    if epoch:
        cp = os.path.join(output_dir, name + str(epoch))
    else:
        cp = os.path.join(output_dir, name)

    torch.save(model.state_dict(), cp)
    return cp

def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, 'module') else model

class ArmijoSGD(Optimizer):
    """Implements ArmijoSGD algorithm, heavily inspired by `minFunc
    <https://en.wikipedia.org/wiki/Backtracking_line_search#CITEREFArmijo1966>`_.


    Args:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(self,
                 params,
                 lr=1,
                 max_iter=1000,
                 tau=.5,
                 c=.5):
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            tau=tau,
            c=c)
        super(ArmijoSGD, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self.curr_lr = self.param_groups[0]['lr']

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        tau = group['tau']
        c = group['c']

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        # f(x): original loss value
        orig_loss_float = float(orig_loss)

        # p: descent direction is negative gradient
        descent_direction = -1*self._params[0].grad.detach().clone()
        # m: negative norm of gradient
        m = torch.dot(descent_direction, self._params[0].grad)
        # t: product of hyperparameter, c , and m
        t = -1*c*m

        x_init = self._clone_param()[0]
        x = self._params[0]

        n_iter = 0
        while n_iter < max_iter:
            # keep track of nb of iterations
            
            alpha = self.curr_lr*((1/tau)**n_iter)

            # Gradient step
            x.copy_(x_init + alpha*descent_direction)

            # Updated loss
            loss = closure()
            
            ############################################################
            # check conditions
            ############################################################

            # Armijo condition
            if orig_loss_float - float(loss) < alpha*t or alpha > lr:
                self.curr_lr = alpha*tau
                break

            n_iter += 1
        
        n_iter = 0
        while n_iter < max_iter:
            # keep track of nb of iterations
            
            alpha = self.curr_lr*(tau**n_iter)

            # Gradient step
            x.copy_(x_init + alpha*descent_direction)

            # Updated loss
            loss = closure()

            # import pdb; pdb.set_trace()
            
            ############################################################
            # check conditions
            ############################################################

            # Armijo condition
            if orig_loss_float - float(loss) >= alpha*t:
                self.curr_lr = alpha
                break

            # lack of progress
            # if d.mul(t).abs().max() <= tolerance_change:
            #     break

            # if abs(loss - prev_loss) < tolerance_change:
            #     break
            n_iter += 1

        return orig_loss