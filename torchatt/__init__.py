# None attacks
from .attacks.vanila import VANILA
from .attacks.gn import GN

# Linf attacks
from .attacks.fgsm import FGSM
from .attacks.bim import BIM
from .attacks.rfgsm import RFGSM
from .attacks.modify_pgd import PGD
from .attacks.eotpgd import EOTPGD
from .attacks.ffgsm import FFGSM
from .attacks.modify_tpgd import TPGD
from .attacks.modify_mifgsm import MIFGSM
from .attacks.upgd import UPGD
from .attacks.apgd import APGD
from .attacks.apgdt import APGDT
from .attacks.mod_difgsm import DIFGSM
from .attacks.tifgsm import TIFGSM
from .attacks.jitter import Jitter
from .attacks.modify_nifgsm import NIFGSM
from .attacks.pgdrs import PGDRS
from .attacks.sinifgsm import SINIFGSM
from .attacks.vmifgsm import VMIFGSM
from .attacks.vnifgsm import VNIFGSM

# L2 attacks
from .attacks.cw import CW
from .attacks.pgdl2 import PGDL2
from .attacks.pgdrsl2 import PGDRSL2
from .attacks.deepfool import DeepFool

# L0 attacks
from .attacks.sparsefool import SparseFool
from .attacks.onepixel import OnePixel
from .attacks.pixle import Pixle

# Linf, L2 attacks
from .attacks.fab import FAB
from .attacks.autoattack import AutoAttack
from .attacks.square import Square

# Wrapper Class
from .wrappers.multiattack import MultiAttack
from .wrappers.lgv import LGV

__version__ = '3.3.0'
__all__ = [
    "VANILA", "GN",

    "FGSM", "BIM", "RFGSM", "PGD", "EOTPGD", "FFGSM",
    "TPGD", "MIFGSM", "UPGD", "APGD", "APGDT", "DIFGSM",
    "TIFGSM", "Jitter", "NIFGSM", "PGDRS", "SINIFGSM",
    "VMIFGSM", "VNIFGSM",

    "CW", "PGDL2", "DeepFool", "PGDRSL2",

    "SparseFool", "OnePixel", "Pixle",

    "FAB", "AutoAttack", "Square",

    "MultiAttack", "LGV",
]
__wrapper__ = [
    "LGV", "MultiAttack",
]