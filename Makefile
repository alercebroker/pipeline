MODULE_NAME=paps
LIB_DIR = $(MODULE_NAME)/lib

default: $(MODULE_NAME)

$(MODULE_NAME): setup.py $(MODULE_NAME)/functions.pyx $(LIB_DIR)/libfunctions.a
	python3 setup.py build_ext --inplace && rm -f functions.c && rm -Rf build

$(LIB_DIR)/libfunctions.a:
	make -C $(LIB_DIR) libfunctions.a

bdist_wheel: setup.py $(MODULE_NAME)/functions.pyx $(LIB_DIR)/libfunctions.a
	python3 setup.py bdist_wheel

install: setup.py $(MODULE_NAME)/functions.pyx $(LIB_DIR)/libfunctions.a
	python setup.py bdist_wheel
	pip install dist/paps*.whl
	rm -r dist build paps.egg-info

clean:
	rm *.so   && rm $(LIB_DIR)/*.a
