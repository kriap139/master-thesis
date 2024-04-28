from dataclasses import dataclass


@dataclass
class DatasetDownload:
    name: str
    download_url: str
    view_url: str

ALIASES = {
    "Accelerometer": "accel",
    "wave_energy": "wave_e",
    "delays_zurich_transport": "delays_zurich",
    "fps-in-video-games": "fps",
    "rcv1.binary": "rcv1",
    "ACSIncome": "acsi"
}

DOWNLOADS = [
    DatasetDownload("okcupid-stem", "https://www.openml.org/data/download/22044770/dataset", "https://www.openml.org/search?type=data&status=active&id=42734&sort=runs"),
    DatasetDownload("wave_energy", "https://api.openml.org/data/download/22111839/file22f16310c401a.arff", "https://www.openml.org/search?type=data&status=active&id=44975&sort=runs"),
    DatasetDownload("Accelerometer", "https://archive.ics.uci.edu/static/public/846/accelerometer.zip", "https://archive.ics.uci.edu/dataset/846/accelerometer"),
    DatasetDownload("higgs", "https://archive.ics.uci.edu/static/public/280/higgs.zip", "https://archive.ics.uci.edu/dataset/280/higgs"),
    DatasetDownload("COMET_MC", "https://www.openml.org/data/download/1836051/php7Exgyq", "https://www.openml.org/search?type=data&status=active&id=5889&sort=runs"),
    DatasetDownload("delays_zurich_transport", "https://api.openml.org/data/download/22111943/dataset", "https://www.openml.org/search?type=data&status=active&id=45045&sort=runs"),
    DatasetDownload("fps-in-video-games", "https://www.openml.org/data/download/22044773/fps-in-video-games.arff", "https://www.openml.org/search?type=data&status=active&id=42737"),
    DatasetDownload("rcv1.binary", "https://www.openml.org/data/download/1594041/php7t4FlC", "https://www.openml.org/search?type=data&status=active&id=1577&sort=runs"),
    DatasetDownload("ACSIncome", "https://www.openml.org/data/download/22101666/ACSIncome_state_number.arff", "https://www.openml.org/search?type=data&status=active&id=43141&sort=runs"),
    DatasetDownload("electricity", "https://api.openml.org/data/download/22111904/dataset", "https://www.openml.org/search?type=data&status=active&id=45018"),
    DatasetDownload("el_nino", "https://archive.ics.uci.edu/static/public/122/el+nino.zip", "https://archive.ics.uci.edu/dataset/122/el+nino")
    
]