# Turbo-Fats [![Build Status](https://travis-ci.com/alercebroker/turbo-fats.svg?token=FuwtsLbsSNgHY1qXBVmB&branch=paper_paula)](https://travis-ci.com/alercebroker/turbo-fats)

Based on the Feature Analysis for Time Series [FATS](https://github.com/isadoranun/FATS) package, adds performance improvements and new features.

ALeRCE extension of the FATS library, based on 
https://github.com/isadoranun/FATS and https://github.com/jonwihl/FATS-2.0, both 
under MIT Licence.

This library is meant for ALeRCE internal use only. If you want to compute 
light curve features please check https://github.com/alercebroker/lc_classifier
which is a more complete and well documented library.

The description of the new features is available in *Alert Classification for
the ALeRCE Broker System: The Light Curve Classifier*
https://arxiv.org/abs/2008.03311

## Installing Turbo Fats

We recommend installing turbofats from this repository
```bash
  git clone https://github.com/alercebroker/turbo-fats.git
  cd turbo-fats
```

To install turbofats we recommend first installing some dependencies
```python
  pip install numpy Cython
```

This dependencies are used by one of turbofats requirements, then we can install the other packages and finally turbofats as development.
```python
  pip install -r requirements.txt
  pip install -e .
```
