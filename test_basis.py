import numpy as np
from mlcvlab.nn.basis import linear, linear_grad, radial, radial_grad

RESULTS = {}

def test_linear():
    x = np.array([
        [10.], 
        [20.], 
        [30.]], dtype='f')
    
    W = np.array([
        [1., 2., 3.],
        [-1., 0., 2.],], dtype='f')
    linear_ = linear(x, W)
    #print("linear : ", linear_)
    exp_linear_ = np.array([
        [140.],
        [ 50.]],dtype='f')
    RESULTS["TEST_LINEAR_1"] = np.allclose(linear_.round(4), exp_linear_)
    

def test_linear_grad():
    x = np.array([
        [10.], 
        [20.], 
        [30.]], dtype='f')
    linear_grad_ = linear_grad(x)
    # print("linear_grad_ : ", linear_grad_)
    exp_linear_grad_ = np.array([
        [10.],
        [20.],
        [30.]],dtype='f')
    RESULTS["TEST_LINEAR_GRAD_1"] = np.allclose(linear_grad_.round(4), exp_linear_grad_)


def test_radial():
    y = np.array([
        [10., 20, 30.]], dtype='f')
    
    y_hat = np.array([
        [10., 21., 29.]], dtype='f')
    radial_ = radial(y, y_hat)
    exp_radial_ = 2.0
    print("radial : ", radial_)
    RESULTS["TEST_RADIAL_1"] = np.allclose(radial_.round(4), exp_radial_)

def test_radial_grad():
    x = np.array([
        [10.], 
        [20.], 
        [30.]], dtype='f')
    W = np.array([
        [1.],
        [-1.], 
        [0.]], dtype='f') 
    loss_grad_y = 140
    radial_grad_ = radial_grad(loss_grad_y, x, W)
    #print("radial_grad_ : ", radial_grad_)
    exp_radial_grad_ = np.array(
        [[-2520., -5880., -8400.]], dtype='f')
    RESULTS["TEST_RADIAL_GRAD_1"] = np.allclose(radial_grad_.round(4), exp_radial_grad_)

if __name__ == "__main__":
    test_linear()
    test_linear_grad()
    
    test_radial()
    test_radial_grad()

    result =  True
    for k,v in RESULTS.items():
        print(f"{k.rjust(30,' ')} : {str(v).ljust(15,' ')}")
        result = result and v

    print(f"\n\nTEST_BASIS : {result}")
