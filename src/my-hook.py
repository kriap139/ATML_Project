from mmcv.runner import HOOKS, Hook
import glob
import os


@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, work_dir: str, n_keep_models: int = 2):
        self.work_dir = work_dir
        self.n_keep = n_keep_models

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        self.clean_working_dir(self.work_dir, self.n_keep)

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

    @classmethod
    def clean_working_dir(cls, work_dir: str, n_keep = 2):
        models = glob.glob(work_dir + "*.pth")
        models.sort()

        with open(f"{work_dir}/last_checkpoint", mode='r') as f:
            last_fp = " ".join(f.readline()).strip()
        
        models.pop(models.index(os.path.basename(last_fp)))

        n_keep -= 1
        if n_keep < 0:
            pass
        else:
            for _ in range(n_keep):
                models.pop()
        
        for model in models:
            os.remove(model)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True)
    parser.add_argument('--n_keep', type=int, required=False, default=2)

    args = parser.parse_args()
    MyHook.clean_working_dir(args.work_dir, args.n_keep)