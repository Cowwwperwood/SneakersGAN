class SneakersDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        images_dir="images",
        input_size=None,
        target_size=None,
    ):
        self.images_dir = os.path.join(root_dir, images_dir)
        self.input_size = input_size
        self.target_size = target_size
        files = os.listdir(self.images_dir)
        self.all_images = sorted([file for file in files if file.endswith(".jpg")])

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.all_images[idx])
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.input_size is not None:
            input_rgb = cv2.resize(image_rgb, self.input_size)
            input_rgb = (input_rgb / 255).astype(np.float32)
        if self.target_size is not None:
            image_rgb = cv2.resize(image_rgb, self.target_size)
        image_rgb = (image_rgb / 255).astype(np.float32)

        image_rgb = torch.from_numpy(image_rgb * 2 - 1)
        if self.input_size is not None:
            input_rgb = torch.from_numpy(input_rgb * 2 - 1)
            return input_rgb, image_rgb
        else:
            return image_rgb