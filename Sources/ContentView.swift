import SwiftUI

struct ContentView: View {
    @StateObject private var det = Detector()

    var body: some View {
        NavigationStack {
            Form {
                Section("verdict") {
                    Text(det.headline).font(.footnote)
                    Text(det.verdict.text).font(.caption).foregroundStyle(.secondary)
                }

                if !det.vaProbeAvailable {
                    Section("warning") {
                        Text("""
                        MTLResource.gpuAddress needs iOS 18+. VA reuse is not \
                        measurable here, so VA reuses will read 0 and a zero \
                        anomaly count is completely uninformative.
                        """).font(.footnote)
                    }
                }

                Section("self-test (positive control)") {
                    if det.selfTest.isEmpty {
                        Text("not run").font(.caption.monospaced())
                    } else {
                        ForEach(Array(det.selfTest.enumerated()), id: \.offset) {
                            Text($0.element).font(.caption.monospaced())
                        }
                    }
                }

                Section("run") {
                    Text(det.phase).font(.footnote.monospaced())
                    Button(det.running ? "stop" : "start") {
                        if det.running { det.stop() } else { det.start() }
                    }
                    .disabled(!det.ready)
                }

                Section("anomalies") {
                    LabeledContent("write did not land", value: "\(det.counts.vacated)")
                    LabeledContent("stale pattern",       value: "\(det.counts.stale)")
                    LabeledContent("unexplained",         value: "\(det.counts.unexplained)")
                }

                Section("premise probe") {
                    LabeledContent("GPU VA reuses", value: "\(det.probe.vaReuses)")
                    LabeledContent("VA range",      value: String(format: "%.1f MiB", det.probe.rangeMiB))
                    LabeledContent("allocations",   value: "\(det.probe.allocations)")
                }

                Section("informational only") {
                    LabeledContent("max round", value: String(format: "%.1f ms", det.probe.maxMillis))
                    Text("Latency is not evidence the guarded path ran.")
                        .font(.caption).foregroundStyle(.secondary)
                }

                Section("recent") {
                    ForEach(Array(det.recent.enumerated()), id: \.offset) {
                        Text($0.element).font(.caption.monospaced())
                    }
                }

                Section("the one thing this cannot settle") {
                    Text("""
                    Whether Metal buffer map/unmap reaches the SPTM UAT mutation \
                    path is not observable from userland. If AGX sub-allocates \
                    inside a pre-mapped heap, no per-buffer stage-2 edit happens \
                    and this detector can never fire.

                    Only kernel instrumentation or an SRD settles it.
                    """)
                    .font(.footnote)
                }
            }
            .navigationTitle("SPTM race detector")
        }
    }
}

@main
struct RacePoCApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}
