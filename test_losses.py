import numpy as np
from mlcvlab.nn.losses import l2, l2_grad, cross_entropy, cross_entropy_grad

RESULTS = {}

def test_l2():
    y = np.array([
        [10., 20, 30.]], dtype='f')
    
    y_hat = np.array([
        [10., 21., 29.]], dtype='f')
    
    l2_loss = l2(y, y_hat)
    print("l2_loss : ", l2_loss)
    exp_l2_loss = 1.4142135
    RESULTS["TEST_L2"] = np.allclose(l2_loss.round(4), exp_l2_loss)

def test_l2_grad_1():
    y = np.array([
        [10., 20, 30.]], dtype='f')
    
    y_hat = np.array([
        [10., 21., 29.]], dtype='f')
    l2_grad_ = l2_grad(y, y_hat)
    print(f"l2_grad : {l2_grad_}")
    exp_l2_grad = np.array(
        [[ 0.,  -0.70710677,  0.70710677]], dtype='f')
    RESULTS["TEST_L2_GRAD_1"] = np.allclose(l2_grad_.round(8), exp_l2_grad)

def test_cross_entrophy():
    y = np.array([
        [1., 0., 0., 1.]], dtype='f')
    
    y_hat = np.array([
        [0.8 , 0.2, 0.6, 0.9]], dtype='f')
    cross_ent = cross_entropy(y, y_hat).round(8)
    exp_cross_ent = np.array([[0.22314353,0.22314353,0.9162908,0.10536055]])
    # print(cross_ent) 
    RESULTS["TEST_CROSS_ENTROPY"] = np.allclose(cross_ent.round(8), exp_cross_ent) 


# def test_cross_entrophy_grad():
#     #TODO
#     RESULTS["TEST_CROSS_ENTROPHY_GRAD"] = False
#     pass

if __name__ == "__main__":
    test_l2()
    test_l2_grad_1()
    test_cross_entrophy()
    # test_cross_entrophy_grad()

    result =  True
    for k,v in RESULTS.items():
        print(f"{k.rjust(30,' ')} : {str(v).ljust(15,' ')}")
        result = result and v

    print(f"\n\nTEST_LOSSES : {result}")