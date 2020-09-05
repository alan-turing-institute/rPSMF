# Makefile for rPSMF Experiments
#
# Copyright & License: See LICENSE file
#

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-builtin-rules

EXP_DIR_1=./Experiment1
EXP_DIR_3=./Experiment3
EXP_DIR_4=./Experiment4

VENV_DIR=./venv

.PHONY: all

all: Experiment1 Experiment3 Experiment4

################
# Experiment 1 #
################

.PHONY: Experiment1 clean_Experiment1

EXPERIMENT1_OUTPUT_PSMF = $(EXP_DIR_1)/output/paper_plot_psmf_fit.pdf \
			  $(EXP_DIR_1)/output/paper_plot_psmf_bases.pdf \
			  $(EXP_DIR_1)/output/paper_plot_psmf_loss.pdf

EXPERIMENT1_OUTPUT_RPSMF = $(EXP_DIR_1)/output/paper_plot_rpsmf_fit.pdf \
			   $(EXP_DIR_1)/output/paper_plot_rpsmf_bases.pdf \
			   $(EXP_DIR_1)/output/paper_plot_rpsmf_loss.pdf

Experiment1: Experiment1_psmf Experiment1_rpsmf

Experiment1_psmf: $(EXPERIMENT1_OUTPUT_PSMF)

Experiment1_rpsmf: $(EXPERIMENT1_OUTPUT_RPSMF)

$(EXPERIMENT1_OUTPUT_PSMF) &: $(EXP_DIR_1)/synthetic_psmf.py \
	$(EXP_DIR_1)/data.py $(EXP_DIR_1)/tracking.py \
	$(EXP_DIR_1)/psmf.py | venv
	source $(VENV_DIR)/bin/activate && python $< -v -s 35853 \
		--output-fit $(EXP_DIR_1)/output/paper_plot_psmf_fit.pdf \
		--output-bases $(EXP_DIR_1)/output/paper_plot_psmf_bases.pdf \
		--output-cost $(EXP_DIR_1)/output/paper_plot_psmf_cost.pdf

$(EXPERIMENT1_OUTPUT_RPSMF) &: $(EXP_DIR_1)/synthetic_rpsmf.py \
	$(EXP_DIR_1)/data.py $(EXP_DIR_1)/tracking.py \
	$(EXP_DIR_1)/rpsmf.py $(EXP_DIR_1)/psmf.py | venv
	source $(VENV_DIR)/bin/activate && python $< -v -s 35833 \
		--output-fit $(EXP_DIR_1)/output/paper_plot_rpsmf_fit.pdf \
		--output-bases $(EXP_DIR_1)/output/paper_plot_rpsmf_bases.pdf \
		--output-cost $(EXP_DIR_1)/output/paper_plot_rpsmf_cost.pdf

clean_Experiment1:
	rm -f $(EXPERIMENT1_OUTPUT_PSMF)
	rm -f $(EXPERIMENT1_OUTPUT_RPSMF)

################
# Experiment 3 #
################

.PHONY: Experiment3 Experiment3_tables clean_Experiment3

Experiment3: Experiment3_tables

EXP3_PERC=20 30 40
EXP3_METHODS=MLESMF TMF PSMF rPSMF PMF BPMF
EXP3_POLLUTANTS=NO2 PM10 PM25
EXP3_REPEATS=1000
EXP3_TARGETS=

exp3-output-dir:
	mkdir -p $(EXP_DIR_3)/output

exp3-table-dir:
	mkdir -p $(EXP_DIR_3)/tables


## generate rules for experiment 3 for all methods/datasets/missing percentages

define RunLondonAir
EXP3_TARGETS += $(EXP_DIR_3)/output/$(1)_$(2)_$(3).json

$(EXP_DIR_3)/output/$(1)_$(2)_$(3).json: $(EXP_DIR_3)/data/LondonAir_$(1).csv \
	$(EXP_DIR_3)/LondonAir_$(3).py | exp3-output-dir venv
	source $(VENV_DIR)/bin/activate && \
		python $(EXP_DIR_3)/LondonAir_$(3).py -i $$< -o $$@ -p $(2) \
		-s 123 -r $(EXP3_REPEATS) -f
endef

$(foreach perc,$(EXP3_PERC),\
$(foreach method,$(EXP3_METHODS),\
$(foreach pollutant,$(EXP3_POLLUTANTS),\
	$(eval $(call RunLondonAir,$(pollutant),$(perc),$(method)))\
)))

## tables for experiment 3

Experiment3_tables: $(EXP_DIR_3)/tables/table_imputation.tex \
	$(EXP_DIR_3)/tables/table_coverage.tex

$(EXP_DIR_3)/tables/table_%.tex: $(EXP_DIR_3)/LondonAir_table_%.py \
	$(EXP3_TARGETS) | exp3-table-dir
	python $< -i $(EXP_DIR_3)/output -o $@

## clean up for experiment 3

clean_Experiment3:
	rm -rf $(EXP_DIR_3)/output
	rm -rf $(EXP_DIR_3)/tables

################
# Experiment 4 #
################

.PHONY: Experiment4 Experiment4_tables clean_Experiment4

Experiment4: Experiment4_tables

EXP4_PERC=30
EXP4_METHODS=MLESMF TMF PSMF rPSMF PMF BPMF
EXP4_DAYS=20160930_203718
EXP4_REPEATS=100
EXP4_TARGETS=

exp4-output-dir:
	mkdir -p $(EXP_DIR_4)/output

exp4-table-dir:
	mkdir -p $(EXP_DIR_4)/tables

## generate targets for experiment 4

define RunGasSensor
EXP_4_TARGETS += $(EXP_DIR_4)/output/$(1)_$(2)_$(3).json

$(EXP_DIR_4)/output/$(1)_$(2)_$(3).json: $(EXP_DIR_4)/data/GasSensor_$(1).csv \
	$(EXP_DIR_4)/GasSensor_$(3).py | exp4-output-dir venv
	source $(VENV_DIR)/bin/activate && \
		python $(EXP_DIR_4)/GasSensor_$(3).py -i $$< -o $$@ -p $(2) \
		-s 123 -r $(EXP4_REPEATS) -f
endef

$(foreach perc,$(EXP4_PERC),\
$(foreach method,$(EXP4_METHODS),\
$(foreach day,$(EXP4_DAYS),\
	$(eval $(call RunGasSensor,$(day),$(perc),$(method)))\
)))

## tables for experiment 4

Experiment4_tables: $(EXP_DIR_4)/tables/table_imputation.tex \
	$(EXP_DIR_4)/tables/table_coverage.tex

$(EXP_DIR_4)/tables/table_%.tex: $(EXP_DIR_4)/GasSensor_table_%.py \
	$(EXP4_TARGETS) | exp4-table-dir

## clean up for experiment 4

clean_Experiment4:
	rm -rf $(EXP_DIR_4)/output
	rm -rf $(EXP_DIR_4)/tables

##############
# Virtualenv #
##############

.PHONY: venv

venv: $(VENV_DIR)/bin/activate requirements.txt

$(VENV_DIR)/bin/activate: requirements.txt
	test -d $(VENV_DIR) || virtualenv $(VENV_DIR)
	source $(VENV_DIR)/bin/activate && pip install -r ./requirements.txt
	touch $(VENV_DIR)/bin/activate

clean_venv:
	rm -rf $(VENV_DIR)

############
# Clean up #
############

.PHONY: clean

clean: clean_venv clean_Experiment1

