from distutils.core import setup
import setuptools
setup(
    name='pure_ocean_breeze',
    version='1.4.3',
    description='芷琦哥的回测框架',
    author='chenzongwei',
    author_email='17695480342@163.com',
    py_modules=['pure_ocean_breeze','pure_ocean_breeze.pure_ocean_breeze','pure_ocean_breeze.initialize'],
    url='https://github.com/chen-001/pure_ocean_breeze.git',
    license='MIT',
    packages=setuptools.find_packages(),
    requires=[]
)