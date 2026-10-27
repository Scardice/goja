package goja

import "unsafe"

// AllocationKind names the kinds of objects an AllocationTracker hears about.
type AllocationKind uint8

const (
	// AllocationObject is an ordinary object: literals, new Object(),
	// Object.create(), instances built by constructors and classes, and the
	// objects JSON.parse() produces.
	AllocationObject AllocationKind = iota + 1
	// AllocationArray is an Array, dense or sparse.
	AllocationArray
	// AllocationArrayBuffer is an ArrayBuffer. Every typed array, DataView
	// and Node-style Buffer stores its bytes in one.
	AllocationArrayBuffer
)

func (k AllocationKind) String() string {
	switch k {
	case AllocationObject:
		return "object"
	case AllocationArray:
		return "array"
	case AllocationArrayBuffer:
		return "arraybuffer"
	default:
		return "unknown"
	}
}

// AllocationTracker is told about the objects a Runtime creates and about the
// storage they acquire later. It lets a host estimate how much memory the
// script code it runs keeps alive, typically by holding weak pointers to a
// sample of the reported objects and calling Object.ApproximateSize on the
// survivors.
//
// Allocated is called once, right after an object of a tracked kind has been
// created. bytes is the storage it already owns, which is the data of an
// ArrayBuffer and zero otherwise; arrays built from a list of values report
// that list through Grown immediately afterwards.
//
// Grown is called when an object becomes large: an array's element storage is
// reallocated at GrowthReportFloor elements or more, an ordinary object or a
// sparse array reaches a power-of-two number of properties or items at or
// above GrowthReportFloor, or an ArrayBuffer receives new data. bytes is an
// approximation of the storage the object now owns. The same object may be
// reported several times as it keeps growing.
//
// Both methods run on the goroutine that runs the Runtime while the object is
// being built. They must not call into the Runtime, and they are the only
// cost tracking adds to an allocation, so they should return quickly.
type AllocationTracker interface {
	Allocated(o *Object, kind AllocationKind, bytes int)
	Grown(o *Object, kind AllocationKind, bytes int)
}

// GrowthReportFloor is the smallest element, property or item count at which
// growth is reported to an AllocationTracker.
const GrowthReportFloor = 1024

// SetAllocationTracker installs t, replacing any previous tracker; nil removes
// it. Objects created before the call are never reported. It must be called on
// the goroutine that runs the Runtime.
func (r *Runtime) SetAllocationTracker(t AllocationTracker) {
	r.allocTracker = t
}

func (r *Runtime) noteAllocated(o *Object, kind AllocationKind, bytes int) {
	if t := r.allocTracker; t != nil {
		t.Allocated(o, kind, bytes)
	}
}

func (r *Runtime) noteGrown(o *Object, kind AllocationKind, bytes int) {
	if t := r.allocTracker; t != nil {
		t.Grown(o, kind, bytes)
	}
}

// notePropertyGrowth runs after a property name has been appended.
func (o *baseObject) notePropertyGrowth() {
	if n := len(o.propNames); n >= GrowthReportFloor && n&(n-1) == 0 && o.class == classObject && o.val != nil {
		o.val.runtime.noteGrown(o.val, AllocationObject, approxMapBytes(n))
	}
}

// noteGrowth runs after the element storage has been replaced.
func (a *arrayObject) noteGrowth() {
	if c := cap(a.values); c >= GrowthReportFloor {
		a.val.runtime.noteGrown(a.val, AllocationArray, c*approxValueSlot)
	}
}

// noteGrowth runs after an item has been appended.
func (a *sparseArrayObject) noteGrowth() {
	if n := len(a.items); n >= GrowthReportFloor && n&(n-1) == 0 {
		a.val.runtime.noteGrown(a.val, AllocationArray, cap(a.items)*approxSparseItem)
	}
}

// The sizes ApproximateSize works with. They follow the 64-bit layout and are
// deliberately round: the result is an estimate, not an accounting.
const (
	approxValueSlot = 16 // one Value interface in a slice
	approxBoxed     = 8  // a number boxed into a Value interface
	approxMapHeader = 48
	// approxMapGroup is a swiss-table group: eight control bytes and eight
	// slots of a string key and a Value.
	approxMapGroup = 8 + 8*(16+16)
	// approxPayloadScan bounds how many elements or properties are read to
	// estimate the payload held by a large array or object.
	approxPayloadScan = 1024
)

// approxMapBytes estimates a property map with n entries. Go fills the
// groups of a map to at most 7/8, and even a one-entry map owns a group.
func approxMapBytes(n int) int {
	return approxMapHeader + (n+6)/7*approxMapGroup
}

var (
	approxObjectBase   = int(unsafe.Sizeof(Object{}))
	approxMapEntryNode = int(unsafe.Sizeof(mapEntry{}))
	approxSparseItem   = int(unsafe.Sizeof(sparseArrayItem{}))
	approxPropertyBox  = int(unsafe.Sizeof(valueProperty{}))
	approxConcatString = int(unsafe.Sizeof(concatString{}))
)

// ApproximateSize estimates the bytes o owns directly: the object itself, its
// own property table, the element storage of an array, the data of an
// ArrayBuffer, and the strings and boxed numbers it holds directly. Objects it
// references are not included; a string shared by several holders is counted
// by each. Above approxPayloadScan elements or properties the payload is
// extrapolated from an even sample, so the cost stays bounded.
//
// It must be called on the goroutine that runs o's Runtime.
func (o *Object) ApproximateSize() int {
	switch self := o.self.(type) {
	case *baseObject:
		return approxObjectBase + int(unsafe.Sizeof(*self)) + self.ownPropertyBytes()
	case *arrayObject:
		return approxObjectBase + int(unsafe.Sizeof(*self)) + self.baseObject.ownPropertyBytes() +
			cap(self.values)*approxValueSlot + sampledPayload(len(self.values), func(i int) int { return valuePayload(self.values[i]) })
	case *sparseArrayObject:
		return approxObjectBase + int(unsafe.Sizeof(*self)) + self.baseObject.ownPropertyBytes() +
			cap(self.items)*approxSparseItem + sampledPayload(len(self.items), func(i int) int { return valuePayload(self.items[i].value) })
	case *arrayBufferObject:
		return approxObjectBase + int(unsafe.Sizeof(*self)) + self.baseObject.ownPropertyBytes() + cap(self.data)
	case *guardedObject:
		return approxObjectBase + int(unsafe.Sizeof(*self)) + self.baseObject.ownPropertyBytes() + approxMapBytes(len(self.guardedProps))
	default:
		return approxObjectBase
	}
}

func (o *baseObject) ownPropertyBytes() int {
	size := cap(o.propNames) * approxValueSlot
	if o.values != nil {
		size += approxMapBytes(len(o.values))
	}
	if o.symValues != nil {
		size += approxMapBytes(o.symValues.size) + o.symValues.size*approxMapEntryNode
	}
	names := o.propNames
	return size + sampledPayload(len(names), func(i int) int {
		name := names[i]
		return len(name) + valuePayload(o.values[name])
	})
}

// sampledPayload sums payload(i) over n items, reading at most about
// approxPayloadScan of them and scaling up.
func sampledPayload(n int, payload func(i int) int) int {
	if n == 0 {
		return 0
	}
	step := max(n/approxPayloadScan, 1)
	total, read := 0, 0
	for i := 0; i < n; i += step {
		total += payload(i)
		read++
	}
	if read == n {
		return total
	}
	return int(int64(total) * int64(n) / int64(read))
}

func valuePayload(v Value) int {
	switch v := v.(type) {
	case asciiString:
		return len(v)
	case unicodeString:
		return len(v) * 2
	case *importedString:
		return len(v.s) + len(v.u)*2
	case *concatString:
		// A rope may share its parts with other strings; count its length once
		// per holder like any other string.
		return approxConcatString + v.length
	case valueInt, valueFloat:
		return approxBoxed
	case *valueProperty:
		if v.accessor {
			return approxPropertyBox
		}
		return approxPropertyBox + valuePayload(v.value)
	default:
		return 0
	}
}
