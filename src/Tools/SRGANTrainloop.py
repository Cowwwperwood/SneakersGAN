def init_wandbSR():
    wandb.finish()
    wandb.init(project='SRGAN3', config={
        "lr": 0.0002,
        "b1": 0.5,
        "b2": 0.999,
        "epochs": 15,
        "batch_size": 16
    })
    return wandb.config


class SRGAN:
    def __init__(self, generator, discriminator, validation_lr, d_update_freq=5, d_loss_weight=2.0):
        self.generator = generator
        self.discriminator = discriminator
        self.validation_lr = validation_lr
        self.d_update_freq = d_update_freq  
        self.d_loss_weight = d_loss_weight 

    def gan_loss(self, y_hat, y):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(y_hat, y)

    def mse_loss(self, sr_imgs, hr_imgs):
        return nn.MSELoss()(sr_imgs, hr_imgs)

    def train_discriminator(self, lr_imgs, hr_imgs, opt_d, device):
        valid = torch.ones(hr_imgs.size(0), 1).to(device)
        fake = torch.zeros(lr_imgs.size(0), 1).to(device)

        real_loss = self.gan_loss(self.discriminator(hr_imgs), valid)

        fake_imgs = self.generator(lr_imgs)
        
        fake_imgs_resized = torch.nn.functional.interpolate(
            fake_imgs.permute(0, 3, 1, 2), 
            size=hr_imgs.shape[1:3], 
            mode='bilinear', 
            align_corners=False
        ).permute(0, 2, 3, 1)

        fake_loss = self.gan_loss(self.discriminator(fake_imgs_resized.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2

        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        return d_loss.item()

    def train_generator(self, lr_imgs, hr_imgs, opt_g, device):
        sr_imgs = self.generator(lr_imgs)
        
        sr_imgs = torch.nn.functional.interpolate(sr_imgs.permute(0, 3, 1, 2), size=hr_imgs.shape[1:3], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        valid = torch.ones(lr_imgs.size(0), 1).to(device)

        gan_loss = self.gan_loss(self.discriminator(sr_imgs), valid)

        mse_loss = self.mse_loss(sr_imgs, hr_imgs)

        g_loss = 1 * mse_loss + 0.5 * gan_loss
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        return g_loss.item()


    def log_wandb(self, g_loss, d_loss, epoch):
        wandb.log({
            "g_loss": g_loss,
            "d_loss": d_loss,
            "epoch": epoch
        })	 		

    def save_model(self, epoch):
        torch.save(self.generator.state_dict(), f"GENERATORSR_Fix2_epoch_{epoch+1}.pth")
        torch.save(self.discriminator.state_dict(), f"DISCRIMINATORSR_Fix2_epoch_{epoch+1}.pth")

    def on_epoch_end(self, epoch, dataloader, device): 
        lr_imgs, _ = next(iter(dataloader))  
        lr_imgs = lr_imgs.type(torch.FloatTensor).to(device)

        with torch.no_grad():
            sr_imgs = self.generator(lr_imgs).permute(0, 3, 1, 2)  
            grid = torchvision.utils.make_grid(sr_imgs)

            # Логируем изображение
            wandb.log({"generated_images": wandb.Image(grid)})

    def train(self, dataloader, epochs, device, lr=2e-4, betas=(0.5, 0.999)):
        init_wandbSR()
        opt_g = optim.Adam(self.generator.parameters(), lr=lr/2, betas=betas)
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr*2, betas=betas)

        for epoch in range(epochs):
            visualize_samples2(self.generator, epoch, dataloader, device)
            self.generator.train()
            self.discriminator.train()

            g_loss_avg = 0
            d_loss_avg = 0

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for lr_imgs, hr_imgs in progress_bar:
                lr_imgs = lr_imgs.type(torch.FloatTensor).to(device)
                hr_imgs = hr_imgs.type(torch.FloatTensor).to(device)

                # Обновляем дискриминатор несколько раз (например, 5 раз)
                for _ in range(self.d_update_freq):
                    d_loss = self.train_discriminator(lr_imgs, hr_imgs, opt_d, device)

                g_loss = self.train_generator(lr_imgs, hr_imgs, opt_g, device)

                g_loss_avg += g_loss
                d_loss_avg += d_loss

                progress_bar.set_postfix(g_loss=g_loss, d_loss=d_loss)

            g_loss_avg /= len(dataloader)
            d_loss_avg /= len(dataloader)

            self.log_wandb(g_loss_avg, d_loss_avg, epoch)
            visualize_samples2(self.generator, epoch, dataloader, device)

            self.on_epoch_end(epoch, dataloader, device)

            if (epoch + 1) % 3 == 0:
                self.save_model(epoch)

    def train_setup():
          generator = SRGenerator()  
          discriminator = SRDiscriminator()
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          generator.to(device)
          discriminator.to(device)
          dataset = SneakersSRDataset("data", input_size=LOW_RES_SIZE, target_size=HIGH_RES_SIZE) 
          dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
          validation_lr = 2e-4  
          srgan = SRGAN(generator, discriminator, validation_lr)
          epochs = 10  
          srgan.train(dataloader, epochs, device)
