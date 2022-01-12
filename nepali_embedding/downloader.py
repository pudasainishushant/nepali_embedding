import os
import xml.etree.ElementTree as et
import itertools
from urllib.request import urlopen
import sys
import zipfile
from tqdm import tqdm

class Package:
    def __init__(self, id, url, name='', author='', subdir='', **kw):
        self.id = id
        self.name = name
        self.author = author
        self.url = url
        self.subdir = subdir
        
        ext = os.path.splitext(url.split("/")[-1])[1]
        self.filename = os.path.join(self.subdir, self.id + ext)
        
        self.__dict__.update(kw)
        
    @staticmethod
    def fromxml(xml):
        if isinstance(xml, str):
            xml = et.parse(xml)
        for key in xml.attrib:
            xml.attrib[key] = str(xml.attrib[key])
        return Package(**xml.attrib)

class Downloader:
    DEFAULT_URL = 'https://raw.githubusercontent.com/pudasainishushant/nepali_embedding/package_setup/nepali_embedding/index.xml'

    def __init__(self) -> None:
        self._url = self.DEFAULT_URL
        with urlopen(self._url) as xml_file:
            self._tree = et.parse(xml_file)
            self._root = self._tree.getroot()

    def default_download_dir(self):
        """
        Return the directory to which packages will be downloaded by
        default.  This value can be overridden using the constructor,
        or on a case-by-case basis using the ``download_dir`` argument when
        calling ``download()``.
        On Windows, the default download directory is
        ``PYTHONHOME/lib/nltk``, where *PYTHONHOME* is the
        directory containing Python, e.g. ``C:\\Python25``.
        On all other platforms, the default directory is the first of
        the following which exists or which can be created with write
        permission: ``/usr/share/everest_nlp_data``, ``/usr/local/share/everest_nlp_data``,
        ``/usr/lib/everest_nlp_data``, ``/usr/local/lib/everest_nlp_data``, ``~/everest_nlp_data``.
        """
        # Check if we are on GAE where we cannot write into filesystem.
        if "APPENGINE_RUNTIME" in os.environ:
            return

        # On Windows, use %APPDATA%
        if sys.platform == "win32" and "APPDATA" in os.environ:
            homedir = os.environ["APPDATA"]

        # Otherwise, install in the user's home directory.
        else:
            homedir = os.path.expanduser("~/")
            if homedir == "~/":
                raise ValueError("Could not find a default download directory")

        return os.path.join(homedir, "everest_nlp_data")
    
    def _download_package(self, package, dir):
        filename = os.path.join(dir, f'{package.id}.zip')
        infile = urlopen(package.url)
        with open(filename, 'wb') as outfile:
            num_blocks = max(1, int(package.size) / (1024 * 16))
            for block in tqdm(itertools.count(), desc=f'Downloading {package.id}'):
                s = infile.read(1024 * 16)
                outfile.write(s)
                if not s:
                    break
        infile.close()

        print(f'Unziping the package: {package.id}')
        zf = zipfile.ZipFile(filename)
        zf.extractall(dir)

    def download(self, package_name):
        dir = self.default_download_dir()
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        packages = [Package.fromxml(p) for p in self._root.findall("packages/package")]
        # for p in self._root.findall("packages/package"):
        download_all = True if package_name == 'all' else False
        for package in packages:
            if download_all:
                self._download_package(package=package, dir=dir)
            elif package_name == package.id:
                self._download_package(package=package, dir=dir)
                break

_downloader = Downloader()
download = _downloader.download

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option(
        "-m",
        "--model",
        default=False,
        help="Model Name",
    )

    (options, args) = parser.parse_args()

    downloader = Downloader()

    if args:
        downloader.download(options.model)
    else:
        downloader.download('all')
    