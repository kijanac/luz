import luz
import numpy as np
import pytest

def test_placeholder():
    assert True

# def test_linear_regressor_trivial():
#     w = np.array([0])
#     b = np.array([0])
#     x = np.array([0])

#     lr = luz.LinearRegressor(weights=w, bias=b)

#     assert pytest.approx(lr.predict(x=x)) == pytest.approx(np.array([0]))


# def test_linear_regressor():
#     w = np.array(
#         [
#             [0.78404336, 0.18485443, 0.91510049, 0.97315751],
#             [0.97752802, 0.64870685, 0.70561049, 0.89954557],
#             [0.35878954, 0.28203509, 0.69067207, 0.03065181],
#             [0.07413005, 0.75571977, 0.48920832, 0.74073544],
#             [0.59810665, 0.6412405, 0.93915153, 0.05035105],
#         ]
#     )
#     b = np.array([[0.228675], [0.79957423], [0.80112616], [0.44060335], [0.90214172]])
#     x = np.array([[1, 2, 3, 4]]).T

#     lr = luz.LinearRegressor(weights=w, bias=b)

#     assert pytest.approx(lr.predict(x=x)) == pytest.approx(np.array([0]))
