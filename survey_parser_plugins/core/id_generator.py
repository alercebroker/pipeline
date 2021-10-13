
def id_generator(ra: float, dec: float) -> int:
    """

    :param ra: right ascension in degrees
    :param dec: declination in degrees
    :return: alerce id
    """
    # 19 Digit ID - two spare at the end for up to 100 duplicates
    id = 1000000000000000000

    # 2013-11-15 KWS Altered code to fix the negative RA problem
    if ra < 0.0:
        ra += 360.0

    if ra > 360.0:
        ra -= 360.0

    # Calculation assumes Decimal Degrees:
    ra_hh = int(ra / 15)
    ra_mm = int((ra / 15 - ra_hh) * 60)
    ra_ss = int(((ra / 15 - ra_hh) * 60 - ra_mm) * 60)
    ra_ff = int((((ra / 15 - ra_hh) * 60 - ra_mm) * 60 - ra_ss) * 100)

    if dec >= 0:
        h = 1
    else:
        h = 0
        dec = dec * -1

    dec_deg = int(dec)
    dec_mm = int((dec - dec_deg) * 60)
    dec_ss = int(((dec - dec_deg) * 60 - dec_mm) * 60)
    dec_f = int(((((dec - dec_deg) * 60 - dec_mm) * 60) - dec_ss) * 10)

    id += (ra_hh * 10000000000000000)
    id += (ra_mm * 100000000000000)
    id += (ra_ss * 1000000000000)
    id += (ra_ff * 10000000000)

    id += (h * 1000000000)
    id += (dec_deg * 10000000)
    id += (dec_mm * 100000)
    id += (dec_ss * 1000)
    id += (dec_f * 100)

    return id
