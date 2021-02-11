# Makefile for PSMF project
#
# For copyright & license information please see the README file.

SHELL       := bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS   += --no-builtin-rules

EXP_DIR_SYN     = ./ExperimentSynthetic
EXP_DIR_REC     = ./ExperimentRecursive
EXP_DIR_IMPUTE  = ./ExperimentImpute
EXP_DIR_BEIJING = ./ExperimentBeijing

PKG_DIR   = ./pypsmf
PKG_FILES = $(shell find $(PKG_DIR) -type f -iname '*.py')
VENV_DIR  = ./venv

.PHONY: all

all: ExperimentSynthetic ExperimentImpute ExperimentBeijing ExperimentRecursive

############################
# Experiment 1 - Synthetic #
############################

.PHONY: ExperimentSynthetic ExperimentSynthetic_psmf ExperimentSynthetic_rpsmf \
	syn_output

ExperimentSynthetic: ExperimentSynthetic_psmf ExperimentSynthetic_rpsmf

syn_output:
	mkdir -p $(EXP_DIR_SYN)/output

TARGETS_SYN_PSMF = \
	$(EXP_DIR_SYN)/output/psmf_fit.pdf \
	$(EXP_DIR_SYN)/output/psmf_bases.pdf \
	$(EXP_DIR_SYN)/output/psmf_cost_y.pdf \
	$(EXP_DIR_SYN)/output/psmf_cost_theta.pdf

TARGETS_SYN_RPSMF = \
	$(EXP_DIR_SYN)/output/rpsmf_fit.pdf \
	$(EXP_DIR_SYN)/output/rpsmf_bases.pdf \
	$(EXP_DIR_SYN)/output/rpsmf_cost_y.pdf \
	$(EXP_DIR_SYN)/output/rpsmf_cost_theta.pdf

TARGETS_SYN = $(TARGETS_SYN_PSMF) $(TARGETS_SYN_RPSMF)

ExperimentSynthetic_psmf: $(TARGETS_SYN_PSMF)

ExperimentSynthetic_rpsmf: $(TARGETS_SYN_RPSMF)

$(TARGETS_SYN_PSMF) &: \
	$(EXP_DIR_SYN)/synthetic_psmf.py \
	$(EXP_DIR_SYN)/data.py | syn_output venv
	source $(VENV_DIR)/bin/activate && python $< -v -s 35853 \
		--output-fit $(EXP_DIR_SYN)/output/psmf_fit.pdf \
		--output-bases $(EXP_DIR_SYN)/output/psmf_bases.pdf \
		--output-cost-y $(EXP_DIR_SYN)/output/psmf_cost_y.pdf \
		--output-cost-theta $(EXP_DIR_SYN)/output/psmf_cost_theta.pdf

$(TARGETS_SYN_RPSMF) &: \
	$(EXP_DIR_SYN)/synthetic_rpsmf.py \
	$(EXP_DIR_SYN)/data.py | syn_output venv
	source $(VENV_DIR)/bin/activate && python $< -v -s 35833 \
		--output-fit $(EXP_DIR_SYN)/output/rpsmf_fit.pdf \
		--output-bases $(EXP_DIR_SYN)/output/rpsmf_bases.pdf \
		--output-cost-y $(EXP_DIR_SYN)/output/rpsmf_cost_y.pdf \
		--output-cost-theta $(EXP_DIR_SYN)/output/rpsmf_cost_theta.pdf

######################################
# Experiment 1 - Beijing Temperature #
######################################

.PHONY: ExperimentBeijing ExperimentBeijingPeriodic ExperimentBeijingRandom \
	beijing_output

ExperimentBeijing: ExperimentBeijingPeriodic ExperimentBeijingRandom

beijing_output:
	mkdir -p $(EXP_DIR_BEIJING)/output

TARGETS_BEIJING_PERIODIC = \
	$(EXP_DIR_BEIJING)/output/periodic_bases.pdf \
	$(EXP_DIR_BEIJING)/output/periodic_cost.pdf \
	$(EXP_DIR_BEIJING)/output/periodic_fit.pdf

TARGETS_BEIJING_RANDOM = \
	$(EXP_DIR_BEIJING)/output/randomwalk_bases.pdf \
	$(EXP_DIR_BEIJING)/output/randomwalk_cost.pdf \
	$(EXP_DIR_BEIJING)/output/randomwalk_fit.pdf

TARGETS_BEIJING = $(TARGETS_BEIJING_PERIODIC) $(TARGETS_BEIJING_RANDOM)

ExperimentBeijingPeriodic: $(TARGETS_BEIJING_PERIODIC)

ExperimentBeijingRandom: $(TARGETS_BEIJING_RANDOM)

$(TARGETS_BEIJING_PERIODIC) &: \
	$(EXP_DIR_BEIJING)/beijing_psmf.py \
	$(EXP_DIR_BEIJING)/beijing_temperature.csv | beijing_output venv
	source $(VENV_DIR)/bin/activate && \
		python $< -i $(EXP_DIR_BEIJING)/beijing_temperature.csv \
		--figure periodic -v \
		--output-bases $(EXP_DIR_BEIJING)/output/periodic_bases.pdf \
		--output-cost $(EXP_DIR_BEIJING)/output/periodic_cost.pdf \
		--output-fit $(EXP_DIR_BEIJING)/output/periodic_fit.pdf

$(TARGETS_BEIJING_RANDOM) &: \
	$(EXP_DIR_BEIJING)/beijing_psmf.py \
	$(EXP_DIR_BEIJING)/beijing_temperature.csv | beijing_output venv
	source $(VENV_DIR)/bin/activate && \
		python $< -i $(EXP_DIR_BEIJING)/beijing_temperature.csv \
		--figure random_walk -v \
		--output-bases $(EXP_DIR_BEIJING)/output/randomwalk_bases.pdf \
		--output-cost $(EXP_DIR_BEIJING)/output/randomwalk_cost.pdf \
		--output-fit $(EXP_DIR_BEIJING)/output/randomwalk_fit.pdf

############################
# Experiment 1 - Recursive #
############################

.PHONY: ExperimentRecursive ExperimentRecursive_psmf rec_output

ExperimentRecursive: ExperimentRecursive_psmf

rec_output:
	mkdir -p $(EXP_DIR_REC)/output

TARGETS_REC_PSMF = \
	$(EXP_DIR_REC)/output/psmf_fit.pdf \
	$(EXP_DIR_REC)/output/psmf_bases.pdf \
	$(EXP_DIR_REC)/output/psmf_cost_y.pdf \
	$(EXP_DIR_REC)/output/psmf_cost_theta.pdf

TARGETS_REC = $(TARGETS_REC_PSMF)

ExperimentRecursive_psmf: $(TARGETS_REC_PSMF)

$(TARGETS_REC_PSMF) &: \
	$(EXP_DIR_REC)/synthetic_recursive_psmf.py \
	$(EXP_DIR_REC)/data.py | rec_output venv
	source $(VENV_DIR)/bin/activate && python $< -v -s 5535 \
		--output-fit $(EXP_DIR_REC)/output/psmf_fit.pdf \
		--output-bases $(EXP_DIR_REC)/output/psmf_bases.pdf \
		--output-cost-y $(EXP_DIR_REC)/output/psmf_cost_y.pdf \
		--output-cost-theta $(EXP_DIR_REC)/output/psmf_cost_theta.pdf

#############################
# Experiment 3 - Imputation #
#############################

.PHONY: ExperimentImpute impute-output impute-tables

IMPUTE_METHODS=MLESMF TMF PSMF rPSMF PMF BPMF
IMPUTE_DATA=LondonAir_NO2 \
	    LondonAir_PM10 \
	    LondonAir_PM25 \
	    GasSensor_20160930_203718 \
	    sp500_closing_prices
IMPUTE_REPEATS=100
IMPUTE_PERCENTAGE=20 30 40
IMPUTE_METRICS=imputation coverage

impute-output:
	mkdir -p $(EXP_DIR_IMPUTE)/output

impute-tables:
	mkdir -p $(EXP_DIR_IMPUTE)/tables

define ExpImpute
TARGETS_IMPUTE += $(EXP_DIR_IMPUTE)/output/$(1)_$(2)_$(3).json

$(EXP_DIR_IMPUTE)/output/$(1)_$(2)_$(3).json: $(EXP_DIR_IMPUTE)/data/$(1).csv \
	$(EXP_DIR_IMPUTE)/$(3).py | impute-output venv
	source $(VENV_DIR)/bin/activate && \
		python $(EXP_DIR_IMPUTE)/$(3).py -i $$< -o $$@ -p $(2) -s 123 \
		-r $(IMPUTE_REPEATS) -f
endef

define ExpImputeTable
TABLES_IMPUTE += $(EXP_DIR_IMPUTE)/tables/table_$(1)_$(2).tex

$(EXP_DIR_IMPUTE)/tables/table_$(1)_$(2).tex: \
	$(EXP_DIR_IMPUTE)/table_$(1).py $(TARGETS_IMPUTE) | impute-tables venv
	source $(VENV_DIR)/bin/activate && \
		python $$< -i $(EXP_DIR_IMPUTE)/output -o $$@ -p $(2)
endef

$(foreach perc,$(IMPUTE_PERCENTAGE),\
$(foreach method,$(IMPUTE_METHODS),\
$(foreach dataset,$(IMPUTE_DATA),\
$(eval $(call ExpImpute,$(dataset),$(perc),$(method)))\
)))

$(foreach perc,$(IMPUTE_PERCENTAGE),\
$(foreach metric,$(IMPUTE_METRICS),\
$(eval $(call ExpImputeTable,$(metric),$(perc)))\
))

ExperimentImpute: $(TARGETS_IMPUTE) $(TABLES_IMPUTE)

##############
# Virtualenv #
##############

.PHONY: venv

venv: $(VENV_DIR)/bin/activate requirements.txt

$(VENV_DIR)/bin/activate: requirements.txt $(PKG_FILES)
	test -d $(VENV_DIR) || python -m venv $(VENV_DIR)
	source $(VENV_DIR)/bin/activate && \
		pip install -r ./requirements.txt && pip install -e $(PKG_DIR)
	touch $(VENV_DIR)/bin/activate

############
# Clean up #
############

.PHONY: clean clean_venv

check_clean:
	@echo -n "Are you sure? [y/N]" && read ans && [ $$ans == y ]

clean: clean_venv clean_results

clean_results: check_clean
	rm -f $(TARGETS_SYN)
	rm -f $(TARGETS_REC)
	rm -f $(TARGETS_BEIJING)
	rm -f $(TARGETS_IMPUTE)
	rm -f $(TABLES_IMPUTE)

clean_venv:
	rm -rf $(VENV_DIR)
