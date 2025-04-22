install:
	clib install --dev

test:
	@$(CC) $(CFLAGS) test.c -I deps $(LDFLAGS) -o $@
	@./$@

.PHONY: test
