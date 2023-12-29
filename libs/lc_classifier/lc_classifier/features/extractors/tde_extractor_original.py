# Based in the work of Manuel Pavez
import numpy as np
from scipy.optimize import curve_fit


def fleet_model(t, a, w, m_0):
    func = np.exp(w * t) - a * w * t + m_0
    return func


def chi_squared(x, y, yerr, nu, func, popt):
    """
    Calculate the chi-squared value.

    Parameters:
        x (array_like): The independent variable where the data is measured.
        y (array_like): The dependent data.
        yerr (array_like): The error of y.
        nu (int): number of parameters of the function
        func (callable): The model function, f(x, ...). It must take the independent variable as the first argument and the parameters to fit as separate remaining arguments.
        popt (array_like): Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized

    Returns:
        float: The chi-squared value.
    """
    residuals = y - func(x, *popt)
    # normalizar
    chi = np.sum(((residuals / yerr) ** 2) / (len(y) - nu))
    return chi


def select_range(my_list, start_value, end_value):
    start_index = min(range(len(my_list)), key=lambda i: abs(my_list[i] - start_value))
    end_index = min(range(len(my_list)), key=lambda i: abs(my_list[i] - end_value))
    return start_index, end_index


def decay_function(x, m, decay_factor):
    func = m + 2.5 * decay_factor * np.log10(x)
    return func


def decaimiento_fit(lista_LC, lista_MJD, lista_LC_err):
    dec = []
    M = []
    chi_list = []
    transientes_sin_dec = []
    for i in range(len(lista_LC)):
        try:
            minim = np.argmin(lista_LC[i])
            t_d = lista_MJD[i][minim]
            inicio1, final1 = select_range(lista_MJD[i], t_d - 50, t_d + 200)
            mjd = np.array(lista_MJD[i][inicio1:final1])
            mag = np.array(lista_LC[i][inicio1:final1])
            sigma_mag = np.array(lista_LC_err[i][inicio1:final1])
            minim2 = np.argmin(mag)

            guess = [np.max(mag), 1.6]
            parameters = curve_fit(
                decay_function, mjd[minim2:] - (t_d - 40),
                mag[minim2:], p0=guess,
                sigma=sigma_mag[minim2:],
                bounds=([-10, 0], [25, 10]))
            chi = chi_squared(
                mjd[minim2:] - (t_d - 40),
                mag[minim2:],
                sigma_mag[minim2:],
                nu=2,
                func=decay_function,
                popt=parameters)
            chi_list.append(chi)
            dec.append(parameters[1])
            M.append(parameters[0])
        except:
            dec.append(np.nan)
            M.append(np.nan)
            chi_list.append(np.nan)
            transientes_sin_dec.append(i)
    return M, dec, chi_list


def fleet_fit(lista_LC, lista_MJD, lista_LC_err):
    a = []
    w = []
    m0 = []
    chi_list = []
    for i in range(len(lista_LC)):
        try:
            minim = np.argmin(lista_LC[i])
            t_d = lista_MJD[i][minim]
            inicio1, final1 = select_range(lista_MJD[i], t_d - 100, t_d + 300)
            mjd = np.array(lista_MJD[i][inicio1:final1])
            mag = np.array(lista_LC[i][inicio1:final1])
            sigma_mag = np.array(lista_LC_err[i][inicio1:final1])

            guess = [0.6, 0.1, np.mean(mag)]
            parameters = curve_fit(
                fleet_model, mjd - (t_d - 20),
                mag, p0=guess,
                sigma=sigma_mag,
                bounds=([0, 0, np.min(mag) - 1], [10, 10, np.max(mag) + 1]))
            chi = chi_squared(
                mjd - (t_d - 40),
                mag,
                sigma_mag,
                3,
                fleet_model,
                parameters)
            chi_list.append(chi)
            a.append(parameters[0])
            w.append(parameters[1])
            m0.append(parameters[2])
        except:
            a.append(np.nan)
            w.append(np.nan)
            m0.append(np.nan)
            chi_list.append(np.nan)
    return a, w, m0, chi_list


def bin_color_lc(jd1, mag1, err1, jd2, mag2, err2, delta=10):
    minjd = int(np.amin([np.amin(jd1), np.amin(jd2)]))
    maxjd = int(np.amax([np.amax(jd1), np.amax(jd2)])+1)
    fluxf1 = 10**(-0.4*(mag1+48.6))
    errflux1 = 0.4*np.log(10)*(10.0**(-0.4*(mag1+48.6)))*err1
    fluxf2 = 10**(-0.4*(mag2+48.6))
    errflux2 = 0.4*np.log(10)*(10.0**(-0.4*(mag2+48.6)))*err2
    jdfinal = []
    mag1_final = []
    color = []
    errcolor = []
    jdin = minjd
    while jdin < maxjd:
        mask11 = (jd1 >= jdin) & (jd1 < (jdin+delta))
        dates1 = jd1[mask11]
        fluxes1 = fluxf1[mask11]
        errors1 = errflux1[mask11]
        mask22 = (jd2 >= jdin) & (jd2 < (jdin+delta))
        dates2 = jd2[mask22]
        fluxes2 = fluxf2[mask22]
        errors2 = errflux2[mask22]
        jdf = jdin+delta*0.5
        if (len(dates1) > 1) and (len(dates2) > 1):
            '''fluxes2=fluxes2*1e30
            errors2=errors2*1e30
            fluxes1=fluxes1*1e30
            errors1=errors1*1e30'''
            fluxfb1 = np.median(fluxes1)
            fluxfb2 = np.median(fluxes2)
            errf1 = np.std(fluxes1)
            errf2 = np.std(fluxes2)
            jdfinal.append(jdf)
            '''fluxes1=fluxfb1*1e-30
            errors1=errf1*1e-30
            fluxes1=fluxfb2*1e-30
            ererrors2rf2=errf2*1e-30'''
            magfinal1 = -2.5*np.log10(fluxfb1)-48.6
            magerrfinal1 = (2.5/np.log(10))*(errf1/fluxfb1)
            magfinal2 = -2.5*np.log10(fluxfb2)-48.6
            magerrfinal2 = (2.5/np.log(10))*(errf2/fluxfb2)
            colorr = magfinal1-magfinal2
            color.append(colorr)
            errcolor.append(np.sqrt(magerrfinal1**2+magerrfinal2**2))
            mag1_final.append(magfinal1)
        elif (len(dates1) > 1) and (len(dates2) == 1):
            fluxfb1 = np.median(fluxes1)
            fluxfb2 = fluxes2[0]
            errf1 = np.std(fluxes1)
            errf2 = errors2[0]
            jdfinal.append(jdf)
            '''fluxes1=fluxfb1*1e-30
            errors1=errf1*1e-30
            fluxes1=fluxfb2*1e-30
            ererrors2rf2=errf2*1e-30'''
            magfinal1 = -2.5*np.log10(fluxfb1)-48.6
            magerrfinal1 = (2.5/np.log(10))*(errf1/fluxfb1)
            magfinal2 = -2.5*np.log10(fluxfb2)-48.6
            magerrfinal2 = (2.5/np.log(10))*(errf2/fluxfb2)
            colorr = magfinal1-magfinal2
            color.append(colorr)
            errcolor.append(np.sqrt(magerrfinal1**2+magerrfinal2**2))
            mag1_final.append(magfinal1)
        elif (len(dates2) > 1) and (len(dates1) == 1):
            fluxfb2 = np.median(fluxes2)
            fluxfb1 = fluxes1[0]
            errf2 = np.std(fluxes2)
            errf1 = errors1[0]
            jdfinal.append(jdf)
            magfinal1 = -2.5*np.log10(fluxfb1)-48.6
            magerrfinal1 = (2.5/np.log(10))*(errf1/fluxfb1)
            magfinal2 = -2.5*np.log10(fluxfb2)-48.6
            magerrfinal2 = (2.5/np.log(10))*(errf2/fluxfb2)
            colorr = magfinal1-magfinal2
            color.append(colorr)
            errcolor.append(np.sqrt(magerrfinal1**2+magerrfinal2**2))
            mag1_final.append(magfinal1)
        elif (len(dates2) == 1) and (len(dates1) == 1):
            fluxfb2 = fluxes2[0]
            fluxfb1 = fluxes1[0]
            errf2 = errors2[0]
            errf1 = errors1[0]
            jdfinal.append(jdf)
            magfinal1 = -2.5*np.log10(fluxfb1)-48.6
            magerrfinal1 = (2.5/np.log(10))*(errf1/fluxfb1)
            magfinal2 = -2.5*np.log10(fluxfb2)-48.6
            magerrfinal2 = (2.5/np.log(10))*(errf2/fluxfb2)
            colorr = magfinal1-magfinal2
            color.append(colorr)
            errcolor.append(np.sqrt(magerrfinal1**2+magerrfinal2**2))
            mag1_final.append(magfinal1)
        else:
            pass
        jdin += delta
    jdfinal = np.array(jdfinal)
    color = np.array(color)
    errcolor = np.array(errcolor)
    mag1_final = np.array(mag1_final)
    return jdfinal, color, errcolor, mag1_final
