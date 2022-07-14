class Path(object):
    @staticmethod
    def get_path_of(name):
        if name == "kitti":
            return '/zhouzm/Datasets/kitti'
        elif name == 'cityscapes':
            return '/zhouzm/Datasets/NYU_v2/cityscapes'
        else:
            raise NotImplementedError