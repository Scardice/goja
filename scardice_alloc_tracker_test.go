package goja

import (
	"testing"
)

type allocEvent struct {
	o     *Object
	kind  AllocationKind
	bytes int
	grown bool
}

type recordingAllocTracker struct{ events []allocEvent }

func (t *recordingAllocTracker) Allocated(o *Object, kind AllocationKind, bytes int) {
	t.events = append(t.events, allocEvent{o: o, kind: kind, bytes: bytes})
}

func (t *recordingAllocTracker) Grown(o *Object, kind AllocationKind, bytes int) {
	t.events = append(t.events, allocEvent{o: o, kind: kind, bytes: bytes, grown: true})
}

func (t *recordingAllocTracker) find(o *Object, grown bool) (allocEvent, int) {
	var last allocEvent
	count := 0
	for _, e := range t.events {
		if e.o == o && e.grown == grown {
			last = e
			count++
		}
	}
	return last, count
}

func trackedRun(t *testing.T, script string) (*Runtime, *recordingAllocTracker, *Object) {
	t.Helper()
	r := New()
	tracker := &recordingAllocTracker{}
	r.SetAllocationTracker(tracker)
	v, err := r.RunString(script)
	if err != nil {
		t.Fatal(err)
	}
	r.SetAllocationTracker(nil)
	o, _ := v.(*Object)
	if o == nil {
		t.Fatalf("script returned %v, want an object", v)
	}
	return r, tracker, o
}

func TestAllocationTrackerHearsAboutEachTrackedKindOnce(t *testing.T) {
	for _, tc := range []struct {
		name   string
		script string
		kind   AllocationKind
		bytes  int
	}{
		{"object literal", "({a: 1})", AllocationObject, 0},
		{"constructed instance", "function F() { this.x = 1 }; new F()", AllocationObject, 0},
		{"class instance", "class C { constructor() { this.x = 1 } }; new C()", AllocationObject, 0},
		{"Object.create", "Object.create(null)", AllocationObject, 0},
		{"JSON object", `JSON.parse('{"a": [1, 2]}')`, AllocationObject, 0},
		{"array literal", "[1, 2, 3]", AllocationArray, 0},
		{"array method result", "[1, 2, 3].map(x => x * 2)", AllocationArray, 0},
		{"array buffer", "new ArrayBuffer(4096)", AllocationArrayBuffer, 4096},
		{"typed array buffer", "new Uint32Array(256).buffer", AllocationArrayBuffer, 1024},
		{"typed array copy", "new Uint8Array(new Uint16Array(10)).buffer", AllocationArrayBuffer, 0},
		{"transfer", "new ArrayBuffer(64).transfer(128)", AllocationArrayBuffer, 128},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, tracker, o := trackedRun(t, tc.script)
			e, n := tracker.find(o, false)
			if n != 1 {
				t.Fatalf("allocation of the result reported %d times, want once (events %d)", n, len(tracker.events))
			}
			if e.kind != tc.kind || e.bytes != tc.bytes {
				t.Fatalf("reported kind %v bytes %d, want %v %d", e.kind, e.bytes, tc.kind, tc.bytes)
			}
		})
	}
}

func TestAllocationTrackerIgnoresFunctionsAndErrors(t *testing.T) {
	r := New()
	tracker := &recordingAllocTracker{}
	r.SetAllocationTracker(tracker)
	v, err := r.RunString("(function f() {})")
	if err != nil {
		t.Fatal(err)
	}
	if _, n := tracker.find(v.(*Object), false); n != 0 {
		t.Fatal("a function was reported as a tracked allocation")
	}
}

func TestAllocationTrackerHearsWhenObjectsBecomeLarge(t *testing.T) {
	for _, tc := range []struct {
		name   string
		script string
		kind   AllocationKind
		min    int
	}{
		{"pushed array", "const a = []; for (let i = 0; i < 5000; i++) a.push(i); a", AllocationArray, 5000 * approxValueSlot},
		{"array from values", "Array.from({length: 3000}, (_, i) => i)", AllocationArray, 3000 * approxValueSlot},
		{"JSON array", "JSON.parse('[' + Array(2000).fill('1').join(',') + ']')", AllocationArray, 2000 * approxValueSlot},
		{"dictionary object", "const o = {}; for (let i = 0; i < 2048; i++) o['k' + i] = i; o", AllocationObject, 2048 * 32},
		{"sparse array", "const s = []; for (let i = 0; i < 2048; i++) s[i * 100000] = i; s", AllocationArray, 2048 * approxSparseItem},
		{"typed array copy", "new Uint8Array(new Uint16Array(100000)).buffer", AllocationArrayBuffer, 100000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, tracker, o := trackedRun(t, tc.script)
			e, n := tracker.find(o, true)
			if n == 0 {
				t.Fatal("growth was never reported")
			}
			if e.kind != tc.kind || e.bytes < tc.min {
				t.Fatalf("last growth kind %v bytes %d, want %v and at least %d", e.kind, e.bytes, tc.kind, tc.min)
			}
		})
	}
}

func TestAllocationTrackerStaysQuietForSmallGrowth(t *testing.T) {
	_, tracker, o := trackedRun(t, "const a = []; for (let i = 0; i < 500; i++) a.push({i}); const o = {}; for (let i = 0; i < 500; i++) o['k' + i] = i; [a, o]")
	for _, e := range tracker.events {
		if e.grown {
			t.Fatalf("growth below the floor was reported: %v %d", e.kind, e.bytes)
		}
	}
	if _, n := tracker.find(o, false); n != 1 {
		t.Fatal("the result array was not reported")
	}
}

func TestApproximateSizeFollowsTheStorageAnObjectOwns(t *testing.T) {
	r := New()
	size := func(script string) int {
		t.Helper()
		v, err := r.RunString(script)
		if err != nil {
			t.Fatal(err)
		}
		return v.(*Object).ApproximateSize()
	}
	empty := size("({})")
	if empty <= 0 {
		t.Fatalf("empty object size %d", empty)
	}
	if buffer := size("new ArrayBuffer(1 << 20)"); buffer < 1<<20 || buffer > 1<<20+4096 {
		t.Fatalf("1 MiB buffer estimated at %d", buffer)
	}
	numbers := size("Array.from({length: 10000}, (_, i) => i + 0.5)")
	if numbers < 10000*approxValueSlot || numbers > 10000*(approxValueSlot+approxBoxed)*2 {
		t.Fatalf("10000 numbers estimated at %d", numbers)
	}
	strings := size("Array.from({length: 10000}, (_, i) => 'x'.repeat(100) + i)")
	if strings < 10000*100 || strings > 10000*(100+approxValueSlot)*2 {
		t.Fatalf("10000 strings of ~104 bytes estimated at %d", strings)
	}
	dict := size("const d = {}; for (let i = 0; i < 5000; i++) d['key' + i] = 'v'.repeat(50); d")
	if perEntry := approxMapGroup/7 + approxValueSlot + 50 + 10; dict < 5000*(32+50) || dict > 5000*perEntry*2 {
		t.Fatalf("5000-entry dictionary estimated at %d", dict)
	}
	if grown := size("const g = {a: 1}; g.b = 'y'.repeat(1000); g"); grown < empty+1000 {
		t.Fatalf("size did not follow a property added after allocation: %d", grown)
	}
	if fn := size("(function () {})"); fn <= 0 {
		t.Fatalf("function size %d", fn)
	}
}
