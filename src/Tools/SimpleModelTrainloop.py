def init_wandb():
    wandb.finish()
    wandb.init(project='fcGAN3', config={
        "lr": 0.0002,
        "b1": 0.5,
        "b2": 0.999,
        "epochs": 15,
        "batch_size": 128
    })
    return wandb.config

def gan_loss(y_hat, y):
    return nn.BCELoss()(y_hat, y)

def train_generator(generator, discriminator, imgs, z, optimizer_g, device):
    valid = torch.ones(imgs.size(0), 1).to(device)
    generated_imgs = generator(z)
    g_loss = gan_loss(discriminator(generated_imgs), valid)

    optimizer_g.zero_grad()
    g_loss.backward()
    optimizer_g.step()

    return g_loss.item()

def train_discriminator(discriminator, generator, imgs, z, optimizer_d, device):
    valid = torch.ones(imgs.size(0), 1).to(device)
    fake = torch.zeros(imgs.size(0), 1).to(device)
    
    real_loss = gan_loss(discriminator(imgs), valid)
    fake_loss = gan_loss(discriminator(generator(z).detach()), fake)
    
    d_loss = (real_loss + fake_loss) / 2

    optimizer_d.zero_grad()
    d_loss.backward()
    optimizer_d.step()

    return d_loss.item()

def log_wandb(g_loss, d_loss, epoch):
    wandb.log({
        "g_loss": g_loss,
        "d_loss": d_loss,
        "epoch": epoch
    })


    
def save_model(generator, discriminator, epoch):
    torch.save(generator.state_dict(), f"generator_epoch_{epoch+1}.pth")
    torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch+1}.pth")

def train(model, dataloader, epochs, device, config):
    generator, discriminator = model
    optimizer_g = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.b1, config.b2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.b1, config.b2))

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        g_loss_avg = 0
        d_loss_avg = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for imgs in progress_bar:
            imgs = imgs.type(torch.FloatTensor).to(device)
            
            z = torch.randn(imgs.size(0), 100).to(device)

            g_loss = train_generator(generator, discriminator, imgs, z, optimizer_g, device)
            d_loss = train_discriminator(discriminator, generator, imgs, z, optimizer_d, device)

            g_loss_avg += g_loss
            d_loss_avg += d_loss

            progress_bar.set_postfix(g_loss=g_loss, d_loss=d_loss)

        g_loss_avg /= len(dataloader)
        d_loss_avg /= len(dataloader)

        log_wandb(g_loss_avg, d_loss_avg, epoch)
        scheduler_g.step(g_loss_avg)
        scheduler_d.step(d_loss_avg)

        visualize_samples(generator, epoch, device) 

        save_model(generator, discriminator, epoch)

def setup_and_train():
    config = init_wandb()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SneakersDataset(dataset_root, target_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)



    generator = FCGenerator(noise_dim=NOISE_DIM, img_shape=IMAGE_SIZE + (3,)).to(device) 
    discriminator = FCDiscriminator(img_shape=IMAGE_SIZE + (3,)).to(device) 
    train((generator, discriminator), dataloader, config.epochs, device, config)

