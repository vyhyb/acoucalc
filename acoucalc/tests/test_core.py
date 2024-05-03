from acoucalc.core import (
    add_layer,
    calculate_global_transfer_matrix,
    initial_pv,
    surface_impedance,
    pressure_refl_factor,
    absorption_coefficient,
    transmission_coefficient,
    transmitted_pressure,
    reflected_pressure    
)
import numpy as np

p_test = np.ones(10)
v_test = np.zeros(10)*1e-10
pv_test = np.array([p_test, v_test])
tm = np.array([[np.ones(10), np.zeros(10)], [np.zeros(10), np.ones(10)]])

def test_add_layer() -> None:
    pv_new = add_layer(pv_test, tm)
    assert np.all(pv_new[0] == np.ones(10))
    assert np.all(pv_new[1] == np.zeros(10))
    assert pv_new.shape == pv_test.shape

def test_calculate_global_transfer_matrix() -> None:
    tm_new = calculate_global_transfer_matrix([tm, tm])
    assert np.all(tm_new[0][0] == np.ones(10))
    assert np.all(tm_new[0][1] == np.zeros(10))
    assert np.all(tm_new[1][0] == np.zeros(10))
    assert np.all(tm_new[1][1] == np.ones(10))
    assert tm_new.shape == tm.shape

def test_initial_pv() -> None:
    pv = initial_pv(1, 0, np.linspace(0, 10, 10))
    assert np.all(pv[0] == np.ones(10))
    assert np.all(pv[1] == np.zeros(10))
    assert pv.shape == (2, 10)

def test_surface_impedance() -> None:
    surf_imp = surface_impedance(pv_test)
    assert surf_imp.shape == (10,)

def test_pressure_refl_factor() -> None:
    surf_imp = surface_impedance(pv_test)
    refl_factor = pressure_refl_factor(surf_imp)
    assert refl_factor.shape == (10,)

def test_absorption_coefficient() -> None:
    surf_imp = surface_impedance(pv_test)
    refl_factor = pressure_refl_factor(surf_imp)
    abs_coeff = absorption_coefficient(refl_factor)
    assert abs_coeff.shape == (10,)

def test_transmission_coefficient() -> None:
    char_imp_air = 343*1.21
    tc = transmission_coefficient(tm, char_imp_air)
    assert tc.shape == (10,)

def test_transmitted_pressure() -> None:
    char_imp_air = 343*1.21
    tp = transmitted_pressure(p_test, tm, char_imp_air)
    assert tp.shape == (10,)
    assert np.isclose(tp, p_test).all()

def test_reflected_pressure() -> None:
    char_imp_air = 343*1.21
    rp = reflected_pressure(p_test, tm, char_imp_air)
    assert rp.shape == (10,)
    assert np.isclose(rp, np.zeros(10)).all()