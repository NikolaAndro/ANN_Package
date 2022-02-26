import numpy as np

from mlcvlab.nn.activations import relu, relu_grad, sigmoid, sigmoid_grad, softmax, softmax_grad, tanh, tanh_grad

RESULTS = {}


def test_sigmoid_1():
    x = np.array([1])
    sigm = sigmoid(x)
    expected_sigm = [0.73105858]
    #print(f"sigmoid_1 : ",sigm)
    RESULTS["TEST_SIGMOID_1"] = np.allclose(sigm.round(8), expected_sigm)

def test_sigmoid_2():
    x = np.array([3, 2, 1])
    sigm = sigmoid(x)
    expected_sigm = [0.95257413, 0.88079708, 0.73105858]
    #print(f"sigmoid_2 : ",sigm)
    RESULTS["TEST_SIGMOID_2"] = np.allclose(sigm.round(8), expected_sigm)


def test_sigmoid_grad_1():
    x = np.array([1])
    y = sigmoid(x)
    sigm_grad = sigmoid_grad(y)
    expected_sigm_grad = [0.19661193]
    #print(f"sigmoid_grad_1 : ",sigm_grad)
    RESULTS["TEST_SIGMOID_GRAD_1"] = np.allclose(sigm_grad.round(8), expected_sigm_grad)

def test_sigmoid_grad_2():
    x = np.array([3, 2, 1])
    y = sigmoid(x)
    sigm_grad = sigmoid_grad(y)
    expected_sigm_grad = [0.04517666, 0.10499359, 0.19661193]
    #print(f"sigmoid_grad_2 : ",sigm_grad)
    RESULTS["TEST_SIGMOID_GRAD_2"] = np.allclose(sigm_grad.round(8), expected_sigm_grad)

def test_softmax_1():
    x = np.array([
        [7, 4, 5, 1, 0],
        [4, 9, 1, 0 ,5]])
    z = softmax(x)
    exp_softmax = np.array([
        [8.41387525e-01, 4.18902183e-02, 1.13869419e-01, 2.08559116e-03, 7.67246110e-04],
        [6.57032193e-03, 9.75122235e-01, 3.27117067e-04, 1.20339644e-04,1.78599867e-02]
        ], dtype='f')
    # print("test_softmax_1 : ",z)
    RESULTS["TEST_SOFTMAX_1"] = np.allclose(z.round(8), exp_softmax)

def test_softmax_grad_1():
    #softmax value for [1, 3, 5, 7]
    x = np.array([
        [1, 3, 5, 7]])
    sm=softmax(x)
    # exp_sm = [2.14400878e-03, 1.58422012e-02, 1.17058913e-01, 8.64954877e-01]
    # print("sm :", sm)
    sm_grad = softmax_grad(sm)
    # print("softmax grad:",sm_grad)
    exp_sm_grad = np.array([
        [ 2.13941201e-03, -3.39658185e-05, -2.50975338e-04, -1.85447085e-03],
        [-3.39658185e-05,  1.55912258e-02, -1.85447085e-03, -1.37027892e-02],
        [-2.50975338e-04, -1.85447085e-03,  1.03356124e-01, -1.01250678e-01],
        [-1.85447085e-03, -1.37027892e-02, -1.01250678e-01,  1.16807938e-01]], dtype='f')
    RESULTS["TEST_SOFTMAX_GRAD_1"] = np.allclose(sm_grad.round(8), exp_sm_grad)

def test_tanh_1():
    x = np.array([
        [-4, -3],
        [-2, -1],
        [ 0,  1],
        [ 2,  3]], dtype="f")
    z = tanh(x)
    #print("tanh :",z)
    expected_tanh = np.array([
        [-0.9993, -0.9951],
        [-0.964, -0.7616],
        [ 0.,         0.7616],
        [ 0.964,  0.9951]], dtype='f')
    RESULTS["TEST_TANH_1"] = np.allclose(z.round(4), expected_tanh)
    


def test_tanh_grad_1():
    x = np.array([
        [-4, -3],
        [-2, -1],
        [ 0,  1],
        [ 2,  3]], dtype="f")
    output = tanh_grad(x)
    #print("tanh grad :",output)
    expected_tanh_grad = np.array([
        [0.0013, 0.0099],
        [0.0707, 0.42  ],
        [1.    , 0.42  ],
        [0.0707, 0.0099]], dtype="f")
    RESULTS["TEST_TANH_GRAD_1"] = np.allclose(output.round(4), expected_tanh_grad)

def test_relu():
    A = np.array([
        [-4, -3],
        [-2, -1],
        [ 0,  1],
        [ 2,  3]], dtype="f")
    res = relu(A)
    #print(f"relu : {res}")
    expected_relu = np.array([
        [0., 0.],
        [0., 0.],
        [0., 1.],
        [2., 3.]], dtype="f")
    RESULTS["TEST_RELU"] = np.allclose(res, expected_relu)

def test_relu_grad():
    Z = np.array([
        [0., 0.],
        [0., 0.],
        [0., 1.],
        [2., 3.]], dtype='f')
    relu_grad_ = relu_grad(Z)
    #print(f"relu_grad : {relu_grad_}")
    expected_relu_grad = np.array([
        [0., 0.],
        [0., 0.],
        [0., 1.],
        [1., 1.]], dtype="f")
    RESULTS["TEST_RELU_GRAD"] = np.allclose(relu_grad_, expected_relu_grad)


if __name__ == "__main__":
    test_sigmoid_1()
    test_sigmoid_2()

    test_sigmoid_grad_1()
    test_sigmoid_grad_2()
    
    test_softmax_1()
    test_softmax_grad_1()

    test_tanh_1()
    test_tanh_grad_1()

    test_relu()
    test_relu_grad()

    result =  True
    for k,v in RESULTS.items():
        print(f"{k.rjust(30,' ')} : {str(v).ljust(15,' ')}")
        result = result and v

    print(f"\n\nTEST_ACTIVATIONS : {result}")