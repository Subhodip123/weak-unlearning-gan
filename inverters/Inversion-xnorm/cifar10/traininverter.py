import torch
import wandb
from inverter import GANInverter
from generator import DataGenerator
from utils.dcgan import Generator
from utils.helper import recon_orig
from config import config
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
torch.manual_seed(44)

class TrainInverter(object):
    def __init__(
        self,
        data_generator,
        inverter,
        epochs,
        optimizer,
        early_stopping
    ) -> None:
        self.data_generator = data_generator
        self.inverter = inverter
        self.training_epochs = epochs
        self.train_loader, self.val_loader , self.test_tensor = self.data_generator.train_val_test_split()
        self.encoder_optim = optimizer
        self.early_stopping = early_stopping

    def evaluation(
        self,
        data_loader,
        model_best=None,
        epoch=None,
        loss_type="validation",
    ):
        """Given the test data it will give the best models performance/avg loss"""

        # EVALUATION
        if model_best is None:
            # load best performing model
            model_best = torch.load("./results/inversion_xnorm_new.pt")
        model_best.eval()
        loss = 0.0
        N = 0.0
        for indx_batch, data_batch in enumerate(data_loader):
            data_batch = data_batch.to(device=device)
            loss_t = model_best.loss(data_batch)
            loss = loss + loss_t.item()
            # print(loss_t)
            N = N + data_batch.shape[0]
        loss = loss / N
        if epoch is None:
          print(f"FINAL LOSS: loss={loss}")
        else:
          print(f"Total loss at Epoch: {epoch}, {loss_type} loss={loss}")
        return loss

    def training(self):
        """Train the encoder model"""
        # Main loop
        self.inverter.train()
        patience = 0
        for epoch in range(self.training_epochs):
            # TRAINING
            self.inverter.train()
            for indx_batch, input_batch in enumerate(self.train_loader):
                # data from training loader and stack the user ref image
                input_batch = input_batch.to(device)
                # loss and update
                loss = self.inverter.loss(input=input_batch)
                # print(loss)
                self.encoder_optim.zero_grad()
                loss.backward()
                self.encoder_optim.step()
            # Training loss
            loss_train = self.evaluation(
                self.train_loader,
                model_best=self.inverter,
                epoch=epoch,
                loss_type="training",
            )
            print(f"Total loss at Epoch: {epoch}, loss={loss_train}")
            wandb.log({'train_loss': loss_train})
            # Validation loss
            loss_val = self.evaluation(
                self.val_loader,
                model_best=self.inverter,
                epoch=epoch,
            )
            wandb.log({'val_loss': loss_val})
            if epoch == 0:
                torch.save(self.inverter, "./results/inversion_xnorm_new.pt")
                print("saved! at epoch=0")
                best_nll = loss_val
            else:
                if loss_val < best_nll:
                    torch.save(self.inverter, "./results/inversion_xnorm_new.pt")
                    print("saved! at epoch" + str(epoch))
                    best_nll = loss_val
                else:
                    patience +=1
            if patience > self.early_stopping:
                break


def train_inverter(config=None):
    #--------------------Train Inverter---------------------------#
    latent_dim = 100
    #-------------Generate Data-----------------#
    G = Generator(ngpu=1).requires_grad_(False)
    # load weights
    G.load_state_dict(torch.load("/home/ece/Subhodip/UnlearnGAN/weights_gan/dcgan_checkpoints/cifar10/netG_epoch_199.pth"))
    if torch.cuda.is_available():
        G = G.cuda()        
    # OPTIMIZER
    with wandb.init(config=config, project='optimized_cifar10_inverter_new'):
        config = wandb.config
        inverter_model = GANInverter(in_channels=3, latent_dim=latent_dim, gen_net=G, 
                                     last_layer_size=config.last_layer)
        inverter_model = inverter_model.to(device=device)
        optimizer = torch.optim.Adamax(
            inverter_model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2)
        )
        datagen = DataGenerator(pretrained_gan=G, no_samples=51200, batch_size = config.batch_size, 
                                img_dims=(1, 32, 32))
        train_loader, val_loader, test_tensors, = datagen.train_val_test_split()
        trainer = TrainInverter(
            data_generator=datagen,
            inverter=inverter_model,
            epochs=config.epochs,
            optimizer=optimizer,
            early_stopping=config.early_stop
        )
        trainer.training()
        # qualitative evaluation
        recon_orig(device=device, inverter_path="./results/inversion_xnorm_new.pt", generator=G, 
                   real_image_tensor=test_tensors)
    #----------------------------------------------------#

if __name__ == '__main__':
    configuration = config()
    train_inverter(config=configuration)     

