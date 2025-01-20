def visualize_samples(generator, epoch, device):
    generator.eval()  
    with torch.no_grad(): 
        z = torch.randn(10, 100).to(device) 
        sample_imgs = generator(z).cpu().detach() 
        sample_imgs = sample_imgs / 2 + 0.5 
        
        fig, axes = plt.subplots(1, 10, figsize=(20, 2)) 
        for i, ax in enumerate(axes):
            ax.imshow(sample_imgs[i].numpy()) 
            ax.axis('off')
        
        plt.tight_layout()  
        wandb.log({"generated_samples_epoch_{}".format(epoch + 1): wandb.Image(fig)}) 
        plt.close(fig)  


def visualize_samples2(generator, epoch, dataloader, device): #Для SRGAN
    generator.eval()
    with torch.no_grad():
        lr_imgs, _ = next(iter(dataloader))  
        lr_imgs = lr_imgs.to(device)
        
        sr_imgs = generator(lr_imgs).cpu().detach()
        
        sr_imgs = sr_imgs / 2 + 0.5 
        grid = torchvision.utils.make_grid(sr_imgs.permute(0, 3, 1, 2), nrow=5)
        
        fig, ax = plt.subplots(figsize=(12, 6))  
        ax.imshow(grid.permute(1, 2, 0).numpy())  
        ax.axis('off')
        ax.set_title(f"Generated Samples at Epoch {epoch + 1}", fontsize=16)
        
        plt.tight_layout()
        wandb.log({"generated_samples_epoch_{}".format(epoch + 1): wandb.Image(fig)})  
        plt.close(fig)
