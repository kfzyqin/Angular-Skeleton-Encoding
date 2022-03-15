import datetime
import sys
import traceback


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def import_class_2(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                          (class_str,
                           traceback.format_exception(*sys.exc_info())))



def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_current_time():
    currentDT = datetime.datetime.now()
    return str(currentDT.strftime("%Y-%m-%dT%H-%M-%S"))
