import os
import imageio
from PIL import Image
from utils.my_transforms import get_transforms
# from my_transforms import get_transforms

def getVal(client = 'TNBC', type='val', enhancetype = 'G1'):
    transform_test = {
        'to_tensor': 1
    }

    # data transforms
    test_transform = get_transforms(transform_test)

    dir_aa = '/data/user/FedPA/data_for_train/{:s}/images/{:s}'.format(client, type)
    dir_bb = '/data/user/FedPA/data_for_train/{:s}/images/{:s}_Mix'.format(client,type)
    testPath = client.split('-')[0]
    dir_cc = '/data/user/FedPA/data/{:s}/labels_instance'.format(testPath)

    aa_files = [f for f in os.listdir(dir_aa) if f.endswith('.png')]
    combined_lists = []
    data_list = []

    for aa_file in aa_files:
        bb_file = os.path.join(dir_bb, aa_file)
        file_name_without_extension = os.path.splitext(aa_file)[0]
        cc_file_with_label = os.path.join(dir_cc, f"{file_name_without_extension}_label.png")

        aa_file_full = os.path.join(dir_aa, aa_file)

        if os.path.exists(bb_file) and os.path.exists(cc_file_with_label):
            combined_tuple = (aa_file_full, bb_file, cc_file_with_label)
            combined_lists.append(combined_tuple)

            input = Image.open(aa_file_full)
            inputMix = Image.open(bb_file)
            valLabel = imageio.imread(cc_file_with_label)

            input = test_transform((input,))[0].unsqueeze(0)
            inputMix = test_transform((inputMix,))[0].unsqueeze(0)
            data_list.append((input, inputMix, valLabel))

    return data_list
