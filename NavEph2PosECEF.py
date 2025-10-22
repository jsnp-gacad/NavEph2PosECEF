#!/usr/bin/env python3

import sys, os, re
import numpy as np

C = 299792458.0
MU = 3.986005e14
OMEGA_E = 7.2921151467e-5
PRNS = tuple(range(1, 33))
TRACK_COLORS = [
    '#009E49', '#FCD116', '#CE1126', '#0067A5', '#2E2D29', '#00A859', '#F6B40E', '#D21034',
    '#007749', '#FFD200', '#A50021', '#00A3DD', '#F77F00', '#0B6A4A', '#FFC726', '#BA0C2F',
    '#005C53', '#FFDE59', '#006E3D', '#D2272D', '#FFA400', '#0063B2', '#009E4F', '#FFBC42',
    '#8C1B2F', '#004B49', '#F4C430', '#2B7F3A', '#ED1C24', '#00703C', '#FF931E', '#003865'
]
PRNS = (1,2,3)

def _rinex_numbers(line):
    """Extract numeric fields from a RINEX navigation data line."""
    if not line:
        return []
    cleaned = line.replace('D', 'E').replace('d', 'e')
    nums = re.findall(r'[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', cleaned)
    return nums if nums else cleaned.split()

def calendar2mjd(year, month, day, hour, minute, sec):
    """MJD conversion"""
    if month > 2:
        newyear = year
        newmonth = month
    else:
        newyear = year - 1
        newmonth = month + 12
    A = int(newyear / 100.0)
    B = (2 - A) + int(A / 4.0)
    part1 = int(365.25 * newyear)
    part2 = int(30.6001 * (newmonth + 1))
    resjd = (part1 + part2) + day + 1720994.5 + B
    resmjd = resjd + ((hour * 3600.0 + minute * 60.0 + sec) / 86400.0) - 2400000.5
    return resmjd

def nsteffensen(Mk, e, tol=1e-14, max_iter=50):
    """Newton–Steffensen solver"""
    Mk = np.arctan2(np.sin(Mk), np.cos(Mk))
    p = float(Mk)
    for _ in range(max_iter):
        p0 = p
        p1 = Mk + e * np.sin(p0)
        p2 = Mk + e * np.sin(p1)
        dd = abs(p2 - 2.0 * p1 + p0)
        if dd < tol:
            break
        denom = p2 - 2.0 * p1 + p0
        if denom == 0.0:
            break
        p = p0 - (p1 - p0) * (p1 - p0) / denom
        if abs(p - p0) <= tol:
            break
    return p

def parse_rinex_nav_v2(path):
    """Return list of ephemeris dicts. Each dict has all needed fields + derived A, tocmjd, toemjd."""
    ephs = []
    with open(path, 'r', encoding='ascii', errors='ignore') as f:
        for line in f:
            if 'END OF HEADER' in line:
                break
        while True:
            head = f.readline()
            if not head:
                break
            head = head.replace('D','E').replace('d','e')
            if not head.strip():
                continue
            toks = head.split()
            if len(toks) >= 10:
                prn, yy, mm, dd, hh, mn = int(toks[0]), int(toks[1]), int(toks[2]), int(toks[3]), int(toks[4]), int(toks[5])
                ss, af0, af1, af2 = float(toks[6]), float(toks[7]), float(toks[8]), float(toks[9])
            else:
                prn = int(head[0:2])
                yy = int(head[3:5]); mm = int(head[6:8]); dd = int(head[9:11])
                hh = int(head[12:14]); mn = int(head[15:17]); ss = float(head[18:22])
                af0 = float(head[22:41]); af1 = float(head[41:60]); af2 = float(head[60:79])
            L = []
            for _ in range(6):
                s = f.readline()
                if not s: s = ""
                L.append(s.replace('D','E').replace('d','e'))
            tail = f.readline()
            if not tail: tail = ""
            tail = tail.replace('D','E').replace('d','e')

            t1 = _rinex_numbers(L[0]); t2 = _rinex_numbers(L[1]); t3 = _rinex_numbers(L[2])
            t4 = _rinex_numbers(L[3]); t5 = _rinex_numbers(L[4]); t6 = _rinex_numbers(L[5])
            t7 = _rinex_numbers(tail)

            eph = dict(
                prn = prn,
                tocy = yy, tocm = mm, tocd = dd, toch = hh, tocn = mn, tocs = float(ss),
                af0 = float(af0), af1 = float(af1), af2 = float(af2),
                iode = int(float(t1[0])), crs = float(t1[1]), dn = float(t1[2]), M0 = float(t1[3]),
                cuc = float(t2[0]), e = float(t2[1]), cus = float(t2[2]), sqrtA = float(t2[3]),
                toe = float(t3[0]), cic = float(t3[1]), Omega0 = float(t3[2]), cis = float(t3[3]),
                i0 = float(t4[0]), crc = float(t4[1]), omega = float(t4[2]), Omegadot = float(t4[3]),
                idot = float(t5[0]), gpsweek = int(float(t5[2])),
                svacc = float(t6[0]) if len(t6)>0 else 0.0,
                health = float(t6[1]) if len(t6)>1 else 0.0,
                tgd = float(t6[2]) if len(t6)>2 else 0.0,
                iodc = int(float(t6[3])) if len(t6)>3 else int(float(t1[0])),
                tot = float(t7[0]) if len(t7)>0 else float(t3[0]),
                fit = float(t7[1]) if len(t7)>1 else 0.0
            )
            # Derived
            eph['A'] = eph['sqrtA'] * eph['sqrtA']
            tocy_full = 1900 + yy if yy > 50 else 2000 + yy
            eph['tocmjd'] = calendar2mjd(tocy_full, mm, dd, hh, mn, float(ss))
            eph['toemjd'] = 44244 + eph['gpsweek'] * 7.0 + eph['toe'] / 86400.0
            eph['totmjd'] = 44244 + eph['gpsweek'] * 7.0 + eph['tot'] / 86400.0
            ephs.append(eph)
    return ephs

def year_from_two_digits(yy):
    return (1900 + yy) if yy > 50 else (2000 + yy)

def mjd_from_doy_year(doy, yy, sod):
    year = year_from_two_digits(yy)
    return int(365.25 * (year - 1)) + 428 + doy + 1720981.5 + (sod / 86400.0) - 2400000.5

def compute_positions_for_day(per_sat, doy, yy, step=10):
    sat_toe = { prn: np.array([e['toemjd'] for e in per_sat[prn]], dtype=float)
                for prn in PRNS if per_sat.get(prn) }
    for sod in range(0, 86400, step):
        mjd = mjd_from_doy_year(doy, yy, sod)
        for prn in PRNS:
            if prn not in sat_toe:
                continue
            L = per_sat[prn]
            toes = sat_toe[prn]
            idx = int(np.argmin(np.abs(toes - mjd)))
            e = L[idx]
            tk = (mjd - e['toemjd']) * 86400.0
            if abs(tk) > 7200.0:
                continue
            dt = (mjd - e['tocmjd']) * 86400.0
            b = (e['af0'] + e['af1'] * dt + e['af2'] * dt * dt) * C
            a = e['A']
            n0 = np.sqrt(MU / (a ** 3))
            n = n0 + e['dn']
            Mk = e['M0'] + n * tk
            Ek = nsteffensen(Mk, e['e'])
            sinE = np.sin(Ek); cosE = np.cos(Ek)
            nu = np.arctan2(np.sqrt(1.0 - e['e'] * e['e']) * sinE, (cosE - e['e']))
            phi = nu + e['omega']
            two_phi = 2.0 * phi
            duk = e['cus'] * np.sin(two_phi) + e['cuc'] * np.cos(two_phi)
            drk = e['crs'] * np.sin(two_phi) + e['crc'] * np.cos(two_phi)
            dik = e['cis'] * np.sin(two_phi) + e['cic'] * np.cos(two_phi)
            uk = phi + duk
            rk = a * (1.0 - e['e'] * cosE) + drk
            ik = e['i0'] + dik + e['idot'] * tk
            x_orb = rk * np.cos(uk)
            y_orb = rk * np.sin(uk)
            Omgk = e['Omega0'] + (e['Omegadot'] - OMEGA_E) * tk - OMEGA_E * e['toe']
            cosO = np.cos(Omgk); sinO = np.sin(Omgk)
            cosi = np.cos(ik);   sini = np.sin(ik)
            x = x_orb * cosO - y_orb * cosi * sinO
            y = x_orb * sinO + y_orb * cosi * cosO
            z = y_orb * sini
            yield (sod, 'G', prn, x, y, z, b, idx + 1, int(e['iode']),
                   e['sqrtA'], e['af0'], e['af1'], e['af2'],
                   e['M0'], e['dn'], e['e'], e['omega'],
                   e['cus'], e['cuc'], e['crs'], e['crc'], e['cis'], e['cic'],
                   e['i0'], e['idot'], e['Omega0'], e['Omegadot'])

def extract_doy_yy_from_name(fname):
    m = re.search(r'(?:^|[^0-9])(\d{3})\d\.(\d{2})(?:[^0-9]|$)', fname)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def ecef_to_latlon_deg(x, y, z):
    """Spherical Earth approximation (sufficient for ground-track visualization)."""
    rxy = np.sqrt(x*x + y*y)
    lat = np.degrees(np.arctan2(z, rxy))
    lon = np.degrees(np.arctan2(y, x))
    # wrap lon to [-180, 180]
    lon = (lon + 180.0) % 360.0 - 180.0
    return lat, lon

def read_brdc_dat(path):
    """
    Reads BRDC_*.txt produced by this script.
    Returns a dict: { prn: {'sod':[], 'lat':[], 'lon':[]} }
    """
    data = {}
    with open(path, 'r', encoding='ascii', errors='ignore') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            sod = int(parts[0])
            prn = int(parts[2])
            x = float(parts[3]); y = float(parts[4]); z = float(parts[5])
            lat, lon = ecef_to_latlon_deg(x, y, z)
            d = data.setdefault(prn, {'sod':[], 'lat':[], 'lon':[]})
            d['sod'].append(sod); d['lat'].append(float(lat)); d['lon'].append(float(lon))
    # sort by SOD for each PRN
    for prn in data:
        idx = np.argsort(np.array(data[prn]['sod']))
        for key in ('sod','lat','lon'):
            arr = np.array(data[prn][key])[idx]
            data[prn][key] = arr.tolist()
    return data

def load_world_map_segments():
    """Load world map polylines from world_map2.txt if available."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, 'world_map.txt'),
    ]
    segments = []
    for cand in candidates:
        if not os.path.exists(cand):
            continue
        lon_seg, lat_seg = [], []
        with open(cand, 'r', encoding='ascii', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if lon_seg:
                        segments.append((np.array(lon_seg), np.array(lat_seg)))
                        lon_seg, lat_seg = [], []
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                lat, lon = float(parts[0]), float(parts[1])
                lat_seg.append(lat)
                lon_seg.append(lon)
        if lon_seg:
            segments.append((np.array(lon_seg), np.array(lat_seg)))
        if segments:
            return segments
    return segments

def plot_ground_tracks_from_dat(dat_path, png_out=None):
    """
    Reproduces the idea of the original gnuplot: plot lon (x) vs lat (y) traces for all PRNs.
    One figure, multiple traces.
    """
    import matplotlib.pyplot as plt
    tracks = read_brdc_dat(dat_path)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    # ax.set_facecolor('#f7f7f7')
    ax.grid(linewidth=1.0)
    ax.set_axisbelow(True)
    # Draw world coastlines if available
    for lon_seg, lat_seg in load_world_map_segments():
        ax.plot(lon_seg, lat_seg, color="#909090", linewidth=0.6, alpha=0.8, zorder=1)
    handles = []
    for idx, prn in enumerate(PRNS):
        if prn not in tracks:
            continue
        d = tracks[prn]
        color = TRACK_COLORS[idx % len(TRACK_COLORS)]
        h = ax.scatter(d['lon'], d['lat'], s=2, marker='o', linewidths=0,
                       color=color, alpha=0.85, zorder=2, label=f'G{prn:02d}')
        handles.append(h)
    ax.set_xlabel('Longitude [°]')
    ax.set_ylabel('Latitude [°]')
    ax.set_title('Ground Tracks')
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    legend = ax.legend(handles=handles, loc="upper center",
                       fontsize='small', bbox_to_anchor=(0.5, -0.16),
                       ncol=8, frameon=False, scatterpoints=1)
    for handle in legend.legendHandles:
        handle.set_sizes([70])
    if png_out:
        fig.savefig(png_out, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def main(argv):
    if len(argv) >= 2 and argv[1] == '--plot':
        if len(argv) < 3:
            print('Usage: python NavEph2PosECEF.py --plot BRDC_file.txt [output.png]')
            return 1
        dat_path = argv[2]
        png_out = argv[3] if len(argv) >= 4 else None
        plot_ground_tracks_from_dat(dat_path, png_out)
        if png_out:
            print(f'Wrote plot: {png_out}')
        return 0

    if len(argv) != 2:
        print(f'Usage: python {os.path.basename(argv[0])} <RINEX NAV v2 file>')
        print(f'   or: python {os.path.basename(argv[0])} --plot BRDC_file.txt [output.png]')
        return 1

    nav_file = argv[1]
    if not os.path.exists(nav_file):
        print(f'Error: file not found: {nav_file}')
        return 1

    fname = os.path.basename(nav_file)
    fnav_wop = fname.replace('.', '', 1)
    out_path = os.path.join(os.path.dirname(nav_file), f'BRDC_{fnav_wop}.txt')

    doy, yy = extract_doy_yy_from_name(fname)

    ephs = parse_rinex_nav_v2(nav_file)
    if not ephs:
        print('No ephemeris found in file.')
        return 1
    per_sat = {prn: [] for prn in PRNS}
    for eph in ephs:
        per_sat.setdefault(eph['prn'], []).append(eph)

    with open(out_path, 'w', encoding='ascii') as out:
        for row in compute_positions_for_day(per_sat, doy, yy, step=5):
            (sod, sysc, prn, x, y, z, b, eph_idx, IODE,
             sqrta, a0, a1, a2, M0, dn, ecc, omega,
             cus, cuc, crs, crc, cis, cic,
             i0, idot, Omega0, Omegadot) = row
            line = (f"{sod:5d} {sysc} {prn:02d}"
                    f" {x:15.4f} {y:15.4f} {z:15.4f} {b:15.4f}"
                    f" {eph_idx:2d} {IODE:4d} "
                    f" {sqrta:19.12e} {a0:19.12e} {a1:19.12e} {a2:19.12e}"
                    f" {M0:19.12e} {dn:19.12e} {ecc:19.12e} {omega:19.12e}"
                    f" {cus:19.12e} {cuc:19.12e} {crs:19.12e} {crc:19.12e} {cis:19.12e} {cic:19.12e}"
                    f" {i0:19.12e} {idot:19.12e} {Omega0:19.12e} {Omegadot:19.12e}\n")
            out.write(line)
            out.flush()  # ensure the line is persisted before moving on to the next epoch
    print(f'Wrote: {out_path}')
    print('To plot ground tracks:')
    print(f'  python {os.path.basename(argv[0])} --plot {out_path} tracks.png')
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
