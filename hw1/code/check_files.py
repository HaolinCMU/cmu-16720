import os
andrew_id = 'XXX'


def check_file(file):
    if os.path.isfile(file):
        return True
    else:
        print('{} not found!'.format(file))
        return False
    

if ( check_file('../'+andrew_id+'/code/visual_words.py') and \
     check_file('../'+andrew_id+'/code/visual_recog.py') and \
     check_file('../'+andrew_id+'/code/network_layers.py') and \
     check_file('../'+andrew_id+'/code/deep_recog.py') and \
     check_file('../'+andrew_id+'/code/util.py') and \
     check_file('../'+andrew_id+'/code/main.py') and \
     check_file('../'+andrew_id+'/code/trained_system.npz') and \
     check_file('../'+andrew_id+'/code/trained_system_deep.npz') and \
     check_file('../'+andrew_id+'/code/dictionary.npy') and \
     check_file('../'+andrew_id+'/'+andrew_id+'_hw1.pdf') ):
    print('file check passed!')
else:
    print('file check failed!')

#modify file name according to final naming policy
#images should be included in the report
