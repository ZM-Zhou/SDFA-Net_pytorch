class Path(object):
    @staticmethod
    def get_path_of(name):
        if name == "kitti":
            return '/zhouzm/Datasets/kitti'
        elif name == 'cityscapes':
            return '/zhouzm/Datasets/NYU_v2/cityscapes'
        elif name == 'swin':
            return '/zhouzm/Download/swin_tiny_patch4_window7_224.pth'
        else:
            raise NotImplementedError