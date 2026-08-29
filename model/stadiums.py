"""Team home-city coordinates, for the travel-distance feature (design doc
section 10.1, rank 3).

City-level (lat, lon), not stadium-precise - close enough for relative
distance ordering between an away team's home and the fixture venue, which is
what the doc's "one-time lookup table" asks for. Compiled from general
geographic knowledge of each club's home city, not an API pull (no free,
keyless geocoding service was worth adding for a one-time static table).
Two same-city derby teams intentionally share coordinates (their distance is 0,
correctly).

Coverage: all 189 canonical teams in normalise/teams.yaml as of this build.
A team missing here (e.g. a newly promoted side not yet added) falls back to
"no travel adjustment" in model/travel.py, the same graceful-unknown pattern
as model/referee.py.
"""
from __future__ import annotations

# team -> (latitude, longitude)
COORDS: dict[str, tuple[float, float]] = {
    # ---- Premier League / Championship (England, Wales) ----
    "Arsenal": (51.5549, -0.1084), "Aston Villa": (52.5091, -1.8848),
    "Bournemouth": (50.7352, -1.8384), "Brentford": (51.4907, -0.2887),
    "Brighton": (50.8617, -0.0837), "Chelsea": (51.4816, -0.1909),
    "Crystal Palace": (51.3983, -0.0855), "Everton": (53.4388, -2.9663),
    "Fulham": (51.4749, -0.2216), "Ipswich": (52.0553, 1.1450),
    "Leeds": (53.7778, -1.5722), "Liverpool": (53.4308, -2.9608),
    "Man City": (53.4831, -2.2004), "Man United": (53.4631, -2.2913),
    "Newcastle": (54.9756, -1.6217), "Nott'm Forest": (52.9400, -1.1327),
    "Sunderland": (54.9144, -1.3883), "Tottenham": (51.6043, -0.0664),
    "West Ham": (51.5387, -0.0166), "Wolves": (52.5902, -2.1301),
    "Coventry": (52.4483, -1.4954), "Hull": (53.7466, -0.3676),
    "Wrexham": (53.0435, -2.9925), "Birmingham": (52.4758, -1.8683),
    "Middlesbrough": (54.5782, -1.2166), "West Brom": (52.5091, -1.9639),
    "Derby": (52.9148, -1.4477), "Swansea": (51.6422, -3.9351),
    "Norwich": (52.6222, 1.3092), "Burnley": (53.7890, -2.2302),
    "Charlton": (51.4861, 0.0367), "Preston": (53.7726, -2.6883),
    "Cardiff": (51.4728, -3.2027), "Sheffield United": (53.3703, -1.4713),
    "Watford": (51.6499, -0.4013), "Southampton": (50.9058, -1.3910),
    "Millwall": (51.4858, -0.0512), "Bristol City": (51.4400, -2.6204),
    "Portsmouth": (50.7961, -1.0642), "Blackburn": (53.7285, -2.4894),
    "QPR": (51.5094, -0.2323), "Bolton": (53.5786, -2.5350),
    "Lincoln": (53.2359, -0.5389), "Sheffield Weds": (53.4111, -1.5003),
    "Stoke": (52.9884, -2.1754), "Oxford": (51.7168, -1.2081),
    "Rotherham": (53.4302, -1.3568), "Leicester": (52.6204, -1.1422),
    "Luton": (51.8843, -0.4316), "Huddersfield": (53.6543, -1.7679),
    "Plymouth": (50.3883, -4.1517),

    # ---- Scottish Premiership ----
    "Celtic": (55.8497, -4.2058), "Rangers": (55.8534, -4.3092),
    "Hearts": (55.9422, -3.2320), "Hibernian": (55.9614, -3.1658),
    "Aberdeen": (57.1595, -2.0876), "Kilmarnock": (55.6083, -4.4975),
    "Dundee United": (56.4763, -2.9707), "Dundee": (56.4769, -2.9707),
    "St Mirren": (55.8467, -4.4292), "Motherwell": (55.7803, -3.9964),
    "St Johnstone": (56.4082, -3.4468), "Falkirk": (55.9970, -3.7876),
    "Livingston": (55.8951, -3.5228), "Ross County": (57.5940, -4.4234),

    # ---- La Liga ----
    "Real Madrid": (40.4531, -3.6883), "Barcelona": (41.3809, 2.1228),
    "Ath Madrid": (40.4362, -3.5995), "Ath Bilbao": (43.2642, -2.9500),
    "Sevilla": (37.3841, -5.9709), "Betis": (37.3568, -5.9822),
    "Valencia": (39.4747, -0.3583), "Villarreal": (39.9442, -0.1036),
    "Sociedad": (43.3014, -1.9736), "Celta": (42.2119, -8.7355),
    "Alaves": (42.8467, -2.6716), "Osasuna": (42.7967, -1.6367),
    "Getafe": (40.3255, -3.7146), "Espanol": (41.3479, 2.0672),
    "Girona": (41.9631, 2.8266), "Mallorca": (39.5896, 2.6304),
    "Levante": (39.4644, -0.3540), "Las Palmas": (28.1001, -15.4515),
    "Cadiz": (36.5087, -6.2314), "Almeria": (36.8511, -2.4697),
    "Granada": (37.1534, -3.5990), "Leganes": (40.3273, -3.7635),
    "Vallecano": (40.3922, -3.6598), "Valladolid": (41.6423, -4.7519),
    "La Coruna": (43.3623, -8.4173), "Malaga": (36.7139, -4.4258),
    "Santander": (43.4623, -3.8339), "Elche": (38.2622, -0.6836),
    "Oviedo": (43.3623, -5.8536),

    # ---- Bundesliga ----
    "Bayern Munich": (48.2188, 11.6247), "Dortmund": (51.4926, 7.4518),
    "RB Leipzig": (51.3459, 12.3487), "Leverkusen": (51.0382, 7.0023),
    "M'gladbach": (51.1746, 6.3852), "Freiburg": (47.9968, 7.8228),
    "Hoffenheim": (49.2380, 8.8886), "Union Berlin": (52.4571, 13.5687),
    "Ein Frankfurt": (50.0685, 8.6455), "FC Koln": (50.9339, 6.8747),
    "Werder Bremen": (53.0664, 8.8378), "Wolfsburg": (52.4325, 10.8038),
    "Mainz": (49.9846, 8.2247), "Augsburg": (48.3234, 10.8863),
    "Stuttgart": (48.7928, 9.2320), "Bochum": (51.4906, 7.2358),
    "Hamburg": (53.5872, 9.8985), "Heidenheim": (48.6763, 10.1541),
    "Holstein Kiel": (54.3376, 10.1225), "St Pauli": (53.5547, 9.9670),
    "Darmstadt": (49.8564, 8.6570), "Elversberg": (49.3072, 7.1156),
    "Paderborn": (51.7086, 8.7645), "Schalke 04": (51.5547, 7.0676),

    # ---- Serie A ----
    "Milan": (45.4781, 9.1240), "Inter": (45.4781, 9.1240),
    "Juventus": (45.1096, 7.6413), "Torino": (45.0416, 7.6497),
    "Napoli": (40.8279, 14.1930), "Roma": (41.9339, 12.4547),
    "Lazio": (41.9339, 12.4547), "Atalanta": (45.7089, 9.6809),
    "Fiorentina": (43.7808, 11.2822), "Bologna": (44.4939, 11.3092),
    "Sassuolo": (44.5471, 10.7897), "Udinese": (46.0803, 13.2032),
    "Genoa": (44.4162, 8.9524), "Cagliari": (39.2360, 9.1220),
    "Verona": (45.4351, 10.9986), "Empoli": (43.7215, 10.9542),
    "Lecce": (40.3568, 18.1652), "Parma": (44.8015, 10.3306),
    "Como": (45.8081, 9.0852), "Venezia": (45.4371, 12.3326),
    "Monza": (45.5845, 9.2739), "Frosinone": (41.6400, 13.3500),
    "Salernitana": (40.6480, 14.7826), "Cremonese": (45.1329, 10.0227),
    "Pisa": (43.7228, 10.4017),

    # ---- Ligue 1 ----
    "Paris SG": (48.8414, 2.2530), "Paris FC": (48.8195, 2.3517),
    "Marseille": (43.2696, 5.3958), "Lyon": (45.7649, 4.9822),
    "Monaco": (43.7276, 7.4153), "Lille": (50.6119, 3.1301),
    "Nice": (43.7052, 7.1926), "Rennes": (48.1073, -1.7128),
    "Lens": (50.4328, 2.8149), "Strasbourg": (48.5698, 7.7530),
    "Nantes": (47.2559, -1.5253), "Montpellier": (43.6229, 3.8125),
    "Toulouse": (43.5828, 1.4342), "Reims": (49.2489, 4.0210),
    "Le Havre": (49.5124, 0.1918), "Brest": (48.4823, -4.4936),
    "Auxerre": (47.7889, 3.5561), "Angers": (47.4571, -0.5177),
    "Metz": (49.1097, 6.1719), "St Etienne": (45.4608, 4.3903),
    "Clermont": (45.7825, 3.1102), "Lorient": (47.7508, -3.3654),
    "Troyes": (48.3049, 4.0847), "Le Mans": (48.0075, 0.1996),

    # ---- Primeira Liga ----
    "Benfica": (38.7527, -9.1846), "Sp Lisbon": (38.7608, -9.1614),
    "Porto": (41.1617, -8.5836), "Sp Braga": (41.5620, -8.4275),
    "Guimaraes": (41.4386, -8.2955), "Boavista": (41.1617, -8.5975),
    "Famalicao": (41.4084, -8.5219), "Gil Vicente": (41.5388, -8.6151),
    "Casa Pia": (38.7139, -9.1934), "Moreirense": (41.3583, -8.3572),
    "Rio Ave": (41.3522, -8.7431), "Santa Clara": (37.7412, -25.6756),
    "Estoril": (38.7042, -9.3977), "Estrela": (38.7538, -9.2306),
    "Nacional": (32.6669, -16.9241), "Maritimo": (32.6394, -16.9280),
    "Arouca": (40.9298, -8.2510), "Farense": (37.0173, -7.9312),
    "Chaves": (41.7405, -7.4744), "Tondela": (40.5192, -8.0817),
    "Vizela": (41.3814, -8.3067), "AVS": (41.3495, -8.3634),
    "Academico Viseu": (40.6566, -7.9122), "Portimonense": (37.1400, -8.5386),
    "Alverca": (38.8814, -9.0389),
}


def coords(team: str) -> tuple[float, float] | None:
    return COORDS.get(team)
