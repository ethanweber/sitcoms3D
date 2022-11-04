import setuptools
import os

# gather dependencies from requirements.txt
lib_folder = os.path.dirname(os.path.realpath(__file__))
reqquirements_filename = lib_folder + "/requirements.txt"
install_requires = []
if os.path.isfile(reqquirements_filename):
    with open(reqquirements_filename) as f:
        install_requires = f.read().splitlines()
install_requires_n = []
for x in install_requires:
    if x != "" and x[0] != "#":
        install_requires_n.append(x)
install_requires = install_requires_n
# print(install_requires)

setuptools.setup(
    name="sitcoms3D",
    version="0.0.0",
    url="https://github.com/ethanweber/sitcoms3D",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=install_requires,
)