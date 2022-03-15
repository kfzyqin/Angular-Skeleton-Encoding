import shutil
import os


def mv_py_files_to_dir(a_dir, tgt_dir=None):
    if tgt_dir is None:
        py_dir = os.path.join(a_dir, 'py_dir')
    else:
        py_dir = tgt_dir
    if not os.path.exists(py_dir):
        os.makedirs(py_dir)

    # for root, dirs, files in os.walk(a_dir):  # copy files
    #     for file in files:
    #         if file.endswith(".py"):
    #             cp_tgt = os.path.join(root, file)
    #             shutil.copy2(cp_tgt, py_dir)

    for item in os.listdir(a_dir):
        s = os.path.join(a_dir, item)
        d = os.path.join(py_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

