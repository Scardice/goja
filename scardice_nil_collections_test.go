package goja

import "testing"

// Scardice keeps converting nil Go slices and maps that are not settable (for
// example function results) to empty JS collections, as goja did before the
// upstream change that turned them into null. Plugins depend on `.length` and
// iteration working on such results.
func TestScardiceNilSliceAndMapResultsAreEmptyCollections(t *testing.T) {
	vm := New()
	_ = vm.Set("nilList", func() []string { return nil })
	_ = vm.Set("nilMap", func() map[string]int { return nil })
	_ = vm.Set("nilFunc", func() func() { return nil })
	v, err := vm.RunString(`JSON.stringify({l: nilList(), n: nilList().length, m: nilMap(), k: Object.keys(nilMap()).length, f: nilFunc()})`)
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"l":[],"n":0,"m":{},"k":0,"f":null}`
	if got := v.String(); got != want {
		t.Fatalf("got %s, want %s", got, want)
	}
}
