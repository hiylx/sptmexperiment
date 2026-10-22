//
//  Detector.swift
//
//  Attempts to observe a lost SPTM stage-2 (UAT) TLBI from an unprivileged app.
//
//  EXPERIMENTAL DESIGN
//    positive control : deliberate foreign write, proves the checker can fire
//    control phase    : no churn at all, proves the checker does not fire spuriously
//    churn phase      : the actual attempt
//
//  CAN establish:
//    - a GPU write landing in a page the CPU is not looking at, unambiguously,
//      via a CPU sentinel the GPU never writes
//    - that the checker works at all (positive control)
//    - its own false-positive rate (control phase)
//    - whether the allocator recycles GPU virtual addresses, via
//      MTLResource.gpuAddress -- the only part of the premise that is observable
//
//  CANNOT establish:
//    - whether Metal buffer map/unmap reaches the SPTM UAT mutation path.
//      If AGX sub-allocates inside a pre-mapped heap there is no per-buffer
//      stage-2 edit and this can never fire. Settling it needs kernel
//      instrumentation or an SRD, not an app.
//
//  Corpus (writeup section 1): SPTM-820.40.18 ships in 24A446 (27.0.1) and
//  24B5084k (27.2b1) and is buggy; fix is SPTM-820.40.23 in 24B5099f.
//
//  A hit requires ALL of: (a) a GPU VA is recycled, (b) the unmap or remap
//  loses its TLBI, (c) the GPU touches the VA before anything corrects it.
//  Only (a) is measurable here. Note the writeup's 8.33 ms is the gate spin
//  timeout, NOT the race window -- the window is instruction-scale.
//

import Foundation
import Metal

// MARK: - build gate

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
            let digits = b.dropFirst(3).prefix(while: { $0.isNumber })
            if let n = Int(digits) { return n >= 5099 ? .fixed(b) : .vulnerable(b) }
        }
        return .unknown(b)
    }
}

// MARK: - detector

final class Detector: ObservableObject {

    // ---- tuning ----
    private let slots         = 256
    private let bytes         = 16 * 1024     // one arm64 page
    private let recycleBatch  = 8
    private let burst         = 64
    private let burstEvery    = 32
    private let controlRounds = 300
    private let publishEvery  = 0.1

    private let sentinel: UInt32 = 0xDEAD_BEEF
    private let poolTag:  UInt32 = 0xA5A5_0000
    private let burstTag: UInt32 = 0xB6B6_0000

    enum Kind: String {
        case vacated     = "write did not land"
        case stale       = "stale pattern"
        case unexplained = "unexplained"
    }

    struct Counts { var vacated = 0; var stale = 0; var unexplained = 0
                    var total: Int { vacated + stale + unexplained } }

    struct Probe {
        var allocations = 0
        var vaReuses    = 0
        var vaLo: UInt64 = 0
        var vaHi: UInt64 = 0
        var maxMillis   = 0.0
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

    private var dev: MTLDevice?
    private var queue: MTLCommandQueue?
    private var pipe: MTLComputePipelineState?
    private var pool = [MTLBuffer?]()
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
        running = true
        stopFlag = false
        work.async { [weak self] in self?.loop() }
    }

    func stop() { stopFlag = true }

    // MARK: headline

    var headline: String {
        if !vaProbeAvailable {
            return "VA reuse unmeasurable on this OS — premise cannot be probed"
        }
        if counts.total > 0 {
            return "\(counts.total) anomalies — a foreign GPU write was observed"
        }
        if probe.vaReuses == 0 {
            return "0 anomalies, 0 VA reuses — uninformative, the detector could not fire"
        }
        return "0 anomalies across \(probe.vaReuses) VA reuses — no hit; premise still unestablished"
    }

    // MARK: address probe

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

        for i in 0..<slots {
            pool[i] = dev.makeBuffer(length: bytes, options: .storageModeShared)
            if let b = pool[i] { fill(b, with: sentinel) }
            note(va(pool[i]))
        }

        // Control: no churn at all, including no burst. Every buffer keeps its
        // VA and its mapping, so an anomaly here is the checker or the GPU.
        publish(phase: "control (no churn)")
        for _ in 0..<controlRounds {
            if stopFlag { break }
            _ = cycle(churn: false, dev: dev, queue: queue, pipe: pipe)
        }

        publish(phase: "churn")
        while !stopFlag {
            _ = cycle(churn: true, dev: dev, queue: queue, pipe: pipe)
        }
        publish(phase: "idle", done: true)
    }

    // MARK: positive control

    /// Deliberately produce both anomaly classes through the real GPU path and
    /// the real classifier, so a later zero means "the bug did not fire"
    /// rather than "the detector does not work".
    private func runSelfTest(dev: MTLDevice, queue: MTLCommandQueue, pipe: MTLComputePipelineState) {
        var out: [String] = []
        defer { let s = out; DispatchQueue.main.async { self.selfTest = s } }

        guard let a = dev.makeBuffer(length: bytes, options: .storageModeShared),
              let b = dev.makeBuffer(length: bytes, options: .storageModeShared) else {
            out.append("FAIL — allocation"); return
        }
        let tagA = poolTag | (0x10 << 8) | 0x01
        let tagB = poolTag | (0x11 << 8) | 0x02

        // 1. honest stamp: both must verify clean
        fill(a, with: sentinel); fill(b, with: sentinel)
        stamp([(a, tagA), (b, tagB)], dev: dev, queue: queue, pipe: pipe)
        out.append(classify(a, want: tagA) == nil ? "PASS  clean stamp (a)" : "FAIL  clean stamp (a)")
        out.append(classify(b, want: tagB) == nil ? "PASS  clean stamp (b)" : "FAIL  clean stamp (b)")

        // 2. foreign: stamp a's VA with b's value. Must classify .stale and
        //    decode slot 0x11.
        stamp([(a, tagB)], dev: dev, queue: queue, pipe: pipe)
        let f = classify(a, want: tagA)
        if f == .stale, let g = firstWord(a), (g >> 8) & 0xFF == 0x11 {
            out.append("PASS  foreign write detected, source slot decoded")
        } else {
            out.append("FAIL  foreign write — got \(f?.rawValue ?? "clean")")
        }

        // 3. vacated: CPU clobber b with the sentinel after a valid stamp.
        stamp([(a, tagA), (b, tagB)], dev: dev, queue: queue, pipe: pipe)
        fill(b, with: sentinel)
        out.append(classify(b, want: tagB) == .vacated
                   ? "PASS  vacated write detected" : "FAIL  vacated write")
    }

    // MARK: one round

    @discardableResult
    private func cycle(churn: Bool, dev: MTLDevice, queue: MTLCommandQueue,
                       pipe: MTLComputePipelineState) -> Int {
        let t0 = CFAbsoluteTimeGetCurrent()
        roundCounter &+= 1
        let round = roundCounter

        var fresh: [Int] = []
        if churn {
            fresh = recycleSlots(dev)
            for i in fresh { if let b = pool[i] { fill(b, with: sentinel) } }
        }

        // Fresh slots first, so map -> GPU touch has nothing in between.
        let order = fresh + (0..<slots).filter { !fresh.contains($0) }
        var items: [(MTLBuffer, UInt32)] = []
        for i in order {
            if let b = pool[i] { items.append((b, expected(slot: i, round: round))) }
        }
        stamp(items, dev: dev, queue: queue, pipe: pipe)

        var bad = 0
        for i in order {
            guard let b = pool[i] else { continue }
            let want = expected(slot: i, round: round)
            guard let kind = classify(b, want: want) else { continue }
            bad += 1
            record(kind, slot: i, round: round, got: firstWord(b) ?? 0)
        }

        // Burst only in the churn phase, and only once the tight window is
        // closed by the verify above.
        if churn && round % burstEvery == 0 {
            burstChurn(dev: dev, queue: queue, pipe: pipe)
        }

        wProbe.maxMillis = max(wProbe.maxMillis, (CFAbsoluteTimeGetCurrent() - t0) * 1000)
        maybePublish()
        return bad
    }

    // MARK: churn

    private func recycleSlots(_ dev: MTLDevice) -> [Int] {
        var fresh: [Int] = []
        for _ in 0..<recycleBatch {
            let i = Int.random(in: 0..<slots)
            if let old = va(pool[i]) {
                if freedVAs.count > 8192 { freedVAs.removeAll() }
                freedVAs.insert(old)
            }
            pool[i] = nil                                    // release -> unmap
            pool[i] = dev.makeBuffer(length: bytes, options: .storageModeShared)
            if let v = va(pool[i]), freedVAs.contains(v) { wProbe.vaReuses += 1 }
            note(va(pool[i]))
            fresh.append(i)
        }
        return fresh
    }

    /// Touch before drop: an untouched buffer has no cached translation, so
    /// dropping it cannot strand one.
    private func burstChurn(dev: MTLDevice, queue: MTLCommandQueue, pipe: MTLComputePipelineState) {
        autoreleasepool {
            var b: [MTLBuffer] = []
            b.reserveCapacity(burst)
            for _ in 0..<burst {
                if let x = dev.makeBuffer(length: bytes, options: .storageModePrivate) { b.append(x) }
            }
            guard !b.isEmpty else { return }
            var items: [(MTLBuffer, UInt32)] = []
            for (k, x) in b.enumerated() { items.append((x, burstTag | UInt32(k & 0xFF))) }
            stamp(items, dev: dev, queue: queue, pipe: pipe)
            b.removeAll()                                    // unmap
        }
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
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// The single classifier, shared by the self-test and the live loop.
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
        case .vacated:     wCounts.vacated += 1
        case .stale:       wCounts.stale += 1
        case .unexplained: wCounts.unexplained += 1
        }
        let line: String
        if kind == .stale {
            line = "r\(round) slot \(slot): holds slot \((got >> 8) & 0xFF) — foreign write"
        } else {
            line = "r\(round) slot \(slot): \(kind.rawValue) (\(String(got, radix: 16)))"
        }
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
        let c = wCounts, p = wProbe
        DispatchQueue.main.async { [weak self] in self?.counts = c; self?.probe = p }
    }

    private func publish(phase p: String, done: Bool = false) {
        let c = wCounts, pr = wProbe
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.phase = p
            self.counts = c
            self.probe = pr
            if done { self.running = false }
        }
    }
}
