import lightning as L
import torch
from torchmetrics.classification import MulticlassAccuracy
import math
from parameterfree import COCOB
from prodigyopt import Prodigy
from schedulefree import AdamWScheduleFree, SGDScheduleFree
import warnings


class ClampedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, clamping=False, pmin=1e-5, reduction='mean'):
        super().__init__()
        self.clamping=clamping
        self.pmin = pmin
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.loss = torch.nn.NLLLoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = self.log_softmax(input)
        if self.clamping:
            out = torch.clamp(out, min=math.log(self.pmin))
        return self.loss(out, target)
    
class SmoothedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, n_classes, clamping=False, pmin=1e-5, reduction='mean'):
        super().__init__()
        self.clamping=clamping
        self.pmin = pmin
        self.n_classes = n_classes
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss = torch.nn.NLLLoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = self.softmax(input)
        if self.clamping:
            out = (1-self.pmin) * out + (self.pmin / self.n_classes)
        out = torch.log(out)
        return self.loss(out, target)
    
class ClippedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, pmin=2, reduction="mean"):
        super().__init__()
        self.clip_val = pmin
        self.loss = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        new_input = torch.zeros_like(input)
        norms = torch.linalg.norm(input.flatten(1), ord=1, dim=1)
        for i in range(input.shape[0]):
            if norms[i] >= self.clip_val:
                new_input[i] = self.clip_val * input[i].clone() / norms[i]
            else:
                new_input[i] = input[i]

        return self.loss(new_input, target)

class ClassificationModel(L.LightningModule):
    def __init__(self, model, optimizer="Adam", lr=1e-3, momentum=0.95, batch_size=64, pmin=1e-5, clamp_method=None,
                  weight_decay=0.01, lr_scheduler=None, nb_batches=None, huber_delta=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.lr = lr
        self.nb_batches = nb_batches
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.model = model
        self.pmin = pmin
        self.clamp_method = clamp_method
        self.configure_loss(clamping=False, pmin=self.pmin)
        self.metric = MulticlassAccuracy(num_classes=self.model.n_classes).to(self.device)
        self.no_reduction_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.huber_loss = torch.nn.HuberLoss(reduction="mean", delta=huber_delta)
        self.opt = None

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)
        train_acc = self.metric(torch.argmax(y_hat, dim=1), y)
        self.log("train_acc", train_acc)

        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.no_reduction_loss(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)
        validation_loss = self.loss(y_hat, y)
        self.log("validation_loss", validation_loss, prog_bar=True)

        huber_loss_val = self.huber_loss(torch.nn.Softmax(dim=1)(y_hat), torch.nn.functional.one_hot(y, self.model.n_classes).to(torch.float32))
        self.log("validation_huber_loss", huber_loss_val, prog_bar=False)

        validation_acc = self.metric(torch.argmax(y_hat, dim=1), y)
        self.log("validation_acc", validation_acc, prog_bar=True)

        validation_error = 1 - validation_acc
        self.log("validation_error", validation_error)
        return torch.nn.CrossEntropyLoss(reduction='sum')(y_hat, y)


    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)
        test_loss = self.loss(y_hat, y)
        self.log("test_loss", test_loss)

        huber_loss_test = self.huber_loss(torch.nn.Softmax(dim=1)(y_hat), torch.nn.functional.one_hot(y, self.model.n_classes).to(torch.float32))
        self.log("test_huber_loss", huber_loss_test, prog_bar=False)

        test_acc = self.metric(torch.argmax(y_hat, dim=1), y)
        self.log("test_acc", test_acc)
        test_error = 1 - test_acc
        self.log("test_error", test_error)

    def __configure_optimizers(self):
        if self.optimizer == "Adam":
            if self.weight_decay == 0:
                opt = torch.optim.Adam(self.parameters(), lr=self.lr)
            else:
                opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "SGD":
            opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer == "sgdfree":
            opt = SGDScheduleFree(self.parameters(),lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer == "cocob":
            opt = COCOB(self.parameters(), alpha=100, weight_decay=self.weight_decay)
        elif self.optimizer == "prodigy":
            opt = Prodigy(self.parameters(), lr=1., weight_decay=self.weight_decay, slice_p=1)
        elif self.optimizer == "adamfree":
            opt = AdamWScheduleFree(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer} not implemented.")
        
        return opt
    
    def __configure_scheduler(self):
        if self.optimizer == "Adam":
            if self.lr_scheduler is None:
                return self.opt
            elif self.lr_scheduler == "plateau":
                scheduler_dict = {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.opt),
                    "interval": "epoch",
                    "monitor": "validation_error"
                }
                return {"optimizer": self.opt, "lr_scheduler": scheduler_dict}
            else:
                raise NotImplementedError(f"The learning rate scheduler {self.lr_scheduler} is not implemented for Adam yet.")
        elif self.optimizer == "SGD":
            if self.lr_scheduler is None or self.nb_batches is None:
                return self.opt
            elif self.lr_scheduler == "onecycle":
                scheduler_dict = {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        self.opt,
                        0.1,
                        epochs=self.trainer.max_epochs,
                        steps_per_epoch=self.nb_batches,
                    ),
                    "interval": "step",
                }
                return {"optimizer": self.opt, "lr_scheduler": scheduler_dict}
            else:
                raise NotImplementedError(f"The learning rate scheduler {self.lr_scheduler} is not implemented for SGD yet.")
        elif self.optimizer == "cocob":
            if self.lr_scheduler is None:
                return self.opt
            else:
                raise NotImplementedError(f"The learning rate scheduler {self.lr_scheduler} is not implemented for {self.optimizer} yet.")
        elif self.optimizer == "prodigy":
            if self.lr_scheduler is None:
                return self.opt
            else:
                raise NotImplementedError(f"The learning rate scheduler {self.lr_scheduler} is not implemented for {self.optimizer} yet.")
        elif self.optimizer == "adamfree":
            if self.lr_scheduler is None:
                return self.opt
            else:
                raise NotImplementedError(f"The learning rate scheduler {self.lr_scheduler} is not implemented for {self.optimizer} yet.")
        elif self.optimizer == "sgdfree":
            if self.lr_scheduler is None:
                return self.opt
            elif self.lr_scheduler == "onecycle":
                warnings.warn(f"\033[31mThe OneCycle scheduler for SGDFree is defined as quick fix for experiment configs.\033[0m", UserWarning)
                return self.opt
            else:
                raise NotImplementedError(f"The learning rate scheduler {self.lr_scheduler} is not implemented for {self.optimizer} yet.")
        else:
            raise NotImplementedError(f"The configuration of learning scheduler is not defined for {self.optimizer}")
    
    def configure_optimizers(self):
        self.opt = self.__configure_optimizers()
        return self.__configure_scheduler()
    
    def on_train_start(self):
        if (self.optimizer == "adamfree" or self.optimizer == "sgdfree") and self.opt is not None:
            self.opt.train()
        return super().on_train_start()
    
    def on_train_end(self):
        if (self.optimizer == "adamfree" or self.optimizer == "sgdfree") and self.opt is not None:
            self.opt.eval()
        return super().on_train_end()
    
    def on_predict_start(self):
        if (self.optimizer == "adamfree" or self.optimizer == "sgdfree") and self.opt is not None:
            self.opt.eval()
        return super().on_predict_start()
    
    def on_predict_end(self):
        if (self.optimizer == "adamfree" or self.optimizer == "sgdfree") and self.opt is not None:
            self.opt.train()
        return super().on_predict_end()
        
    def on_validation_start(self):
        if (self.optimizer == "adamfree" or self.optimizer == "sgdfree") and self.opt is not None:
            self.opt.eval()
        return super().on_validation_start()
    
    def on_validation_end(self):
        if (self.optimizer == "adamfree" or self.optimizer == "sgdfree") and self.opt is not None:
            self.opt.train()
        return super().on_validation_end()
    
    def on_test_start(self):
        if (self.optimizer == "adamfree" or self.optimizer == "sgdfree") and self.opt is not None:
            self.opt.eval()
        return super().on_test_start()
    
    def on_test_end(self):
        if (self.optimizer == "adamfree" or self.optimizer == "sgdfree") and self.opt is not None:
            self.opt.train()
        return super().on_test_end()

    def configure_loss(self, clamping : bool = False, pmin : float = 1e-5):
        if self.clamp_method == "clamp":
            self.loss = ClampedCrossEntropyLoss(clamping=clamping, pmin=pmin/self.model.n_classes, reduction='mean')
        elif self.clamp_method == "smooth":
            self.loss = SmoothedCrossEntropyLoss(n_classes=self.model.n_classes, clamping=clamping, pmin=pmin, reduction='mean')
        elif self.clamp_method == "clip":
            self.loss = ClippedCrossEntropyLoss(pmin=pmin, reduction='mean')
        elif self.clamp_method is None:
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"The clamping method {self.clamp_method} is not implemented yet.")
    

class ProbClassificationModel(ClassificationModel):
    def __init__(self, model, pbbobj, min_val, max_val, optimizer="Adam", lr=1e-3, momentum=0.95, batch_size=64, pmin=1e-5,
                  clamp_method="clamp", weight_decay=0.01, lr_scheduler=None, nb_batches=None,
                  ):
        super().__init__(model, optimizer, lr, momentum, batch_size, pmin, clamp_method, weight_decay, lr_scheduler, nb_batches)
        self.pbbobj = pbbobj
        self.min_val, self.max_val = min_val, max_val
        self.sampling_mode = True


    def set_sampling_mode(self, sample):
        self.sampling_mode = sample

    def training_step(self, batch, batch_idx):
        self.model.set_sampling_mode(self.sampling_mode)
        loss = super().training_step(batch, batch_idx)

        normalized_loss = (loss - self.min_val) / (self.max_val - self.min_val)
        kl = self.model.compute_kl()
        bound_val = self.pbbobj.bound(normalized_loss, kl, self.pbbobj.n_posterior, None)
        return bound_val
    
    def predict_step(self, batch, batch_idx):
        self.model.set_sampling_mode(self.sampling_mode)
        return super().predict_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        self.model.set_sampling_mode(self.sampling_mode)
        out = super().validation_step(batch, batch_idx)
        self.log("kl_val", self.model.compute_kl()/self.pbbobj.n_posterior, prog_bar=True)
        return out


    def test_step(self, batch, batch_idx):
        self.model.set_sampling_mode(self.sampling_mode)
        super().test_step(batch, batch_idx)
        
