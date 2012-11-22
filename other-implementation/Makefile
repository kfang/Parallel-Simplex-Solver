############################################################################
#                        Parallel Simplex - Makefile                       #
############################################################################

#---------------------------------------------------------------------------
# VARIABLES DEFINITIONS

CC = g++
CCOPTS = -O3 $(addprefix -I, $(SRCDIRS))

SRCDIRS = $(shell find code/ -type d)
CPPFILES = $(shell for x in `find code/ -name '*.cpp'`; do echo `basename $$x`; done;)
OBJDIR = obj

TARGETS = simplex
OBJS = ${CPPFILES:%.cpp=%.o}
LIBS = -lboost_regex
LIBDIRS = -L ../lib -L $(OBJDIR) 

#---------------------------------------------------------------------------
# VPATH SETTINGS

vpath %.h $(SRCDIRS)
vpath %.cpp $(SRCDIRS)
vpath %.o $(OBJDIR)

#---------------------------------------------------------------------------
# MAKE COMMANDS

.PHONY: all
all: $(TARGETS)
	@echo DONE!!!

simplex: $(OBJS)
	@echo linking $(OBJS) $(LIBS)
	@cd $(OBJDIR); $(CC) $(CCOPTS) $(OBJS) $(LIBDIRS) $(LIBS) -o ../$@

%.o: %.cpp
	@echo compiling $<...
	@$(CC) $(CCOPTS) $< -c; mkdir -p $(OBJDIR); mv $@ $(OBJDIR)

.PHONY: test
test: $(TARGETS)
	@echo A test script has not been implemented yet.

.PHONY: graphs
graphs:
	@echo Not implemented yet,

.PHONY: clean
clean:
	@echo -e "Removing..." $(addprefix '\n',$(OBJS) $(TARGETS))
	@find . -name '*.o' -print | xargs rm -f; rm -f $(TARGETS)

.PHONY: loc
loc:
	@echo 'Lines of Code for the Project'
	@echo '-----------------------------'
	@find . -regex '.+\.\(h\|\cpp\)' | grep -v boost | xargs wc -l
