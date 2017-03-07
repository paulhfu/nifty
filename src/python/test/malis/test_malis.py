import nifty
import numpy as np

def toy_data_medium():
    x = np.zeros((11,11,2), dtype = 'float32')
    x[:,5,1] = 1.
    x[5,5,1] = .2
    y = np.zeros((11,11), dtype = 'uint32')
    y[:,:5] = 1
    y[:,5:] = 2
    return x,y

def toy_data_small():
    x = .9 * np.ones((5,5,2), dtype = 'float32')
    x[:,2,1] = .1
    x[2,2,1] = .5
    y = np.zeros((5,5), dtype = 'uint32')
    y[:,:3] = 1
    y[:,3:] = 2
    return x,y


def view_res(x, y, pos_grads, neg_grads):
    from volumina_viewer import volumina_flexible_layer
    pos_grads /= pos_grads.max()
    neg_grads /= neg_grads.max()
    volumina_flexible_layer( [x, y, pos_grads, neg_grads],
                             ['Grayscale', 'RandomColors', 'Red', 'Green'],
                             ['Data', 'Labels', 'Pos-Grads', 'Neg-Grads'])

def test_malis():
    x,y = toy_data_small()
    pos_grads, neg_grads = nifty.malis.malis_gradient(x, y)

    ## view
    #view_res(x, y,
    #        pos_grads.astype('float32'),
    #        neg_grads.astype('float32'))

    ## test
    i_max = neg_grads.shape[0]
    j_max = neg_grads.shape[1]
    for i in xrange(i_max):
        for j in xrange(j_max):
            if i == i_max / 2 and j == j_max / 2:
                assert neg_grads[i,j,0] == 0
                assert neg_grads[i,j,1] > 0
            else:
                assert neg_grads[i,j,0] == 0
                assert neg_grads[i,j,1] == 0

    print "Passed"

if __name__ == '__main__':
    test_malis()
