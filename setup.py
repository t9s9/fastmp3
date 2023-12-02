from setuptools import find_packages, setup, Extension

extra_compile_args = ["-std=c++20", "-fpic", "-Wall", "-O3", "-march=native", "-shared"]  # SSE/NEON support

libmp3 = Extension('fastmp3._libmp3',
                   sources=['./libmp3/libmp3.cpp'],
                   include_dirs=[],
                   language='c++',
                   extra_compile_args=extra_compile_args)

setup(name='fastmp3',
      version='0.1',
      description='FastMP3: Fast MP3 Decoder',
      author='Timothy Schaumlöffel',
      author_email='schaumloeffel@em.uni-frankfurt.de',
      packages=find_packages('src'),
      ext_modules=[libmp3],
      package_dir={'': 'src'},
      install_requires=['numpy'],
      tests_require=['pytest', 'librosa'],
      )
