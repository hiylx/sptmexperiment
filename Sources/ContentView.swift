//
//  RacePoC.swift
//
//  UI for the SPTM stage-2 race detector (Detector.swift v4).
//
//  Build: Xcode -> iOS App -> SwiftUI / Swift, drop both files in, delete the
//  generated ContentView.swift. Deployment target 16.0+. No entitlements.
//  Run as a Release build; Debug overhead distorts the attempt rate.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var det = Detector()

    var body: some View {
        NavigationStack {
            Form {
                sectionVerdict
                if !det.vaProbeAvailable { sectionVAWarning }
                sectionSelfTest
                sectionRun
                sectionAnomalies
                sectionPremise
                sectionRecent
                sectionCaveat
                sectionExport
            }
            .navigationTitle("SPTM race detector")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    // MARK: - verdict

    private var sectionVerdict: some View {
        Section {
            Text(det.headline)
                .font(.callout.weight(.medium))
            Text(det.verdict.text)
                .font(.caption)
                .foregroundStyle(.secondary)
        } header: {
            Text("verdict")
        } footer: {
            Text("The headline refuses to say \"clean\". With no VA reuse it reports uninformative; "
               + "with reuse and no anomalies it reports no hit, not bug absent.")
        }
    }

    // MARK: - va probe warning

    private var sectionVAWarning: some View {
        Section {
            Text("MTLResource.gpuAddress needs iOS 18+")
                .font(.callout.weight(.medium))
            Text("VA reuse is not measurable on this OS, so VA reuses will read 0 and a zero anomaly "
               + "count is completely uninformative.")
                .font(.footnote)
        } header: {
            Text("warning")
        }
    }

    // MARK: - positive control

    private var sectionSelfTest: some View {
        Section {
            if det.selfTest.isEmpty {
                Text("not run")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            } else {
                ForEach(Array(det.selfTest.enumerated()), id: \.offset) { _, line in
                    Text(line)
                        .font(.caption.monospaced())
                        .foregroundStyle(line.hasPrefix("PASS") ? Color.green : Color.red)
                }
            }
        } header: {
            Text("self-test (positive control)")
        } footer: {
            Text("Deliberately produces a foreign write and a vacated write through the real GPU path "
               + "and the real classifier. All PASS means a later zero is the bug not firing, not a "
               + "dead detector.")
        }
    }

    // MARK: - run

    private var sectionRun: some View {
        Section {
            HStack {
                Text("phase")
                Spacer()
                Text(det.phase)
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
            }

            Picker("strategy", selection: $det.strategy) {
                ForEach(Strategy.allCases) { s in
                    Text(s.rawValue).tag(s)
                }
            }
            .pickerStyle(.segmented)
            .disabled(det.running)

            Button(det.running ? "stop" : "start") {
                if det.running { det.stop() } else { det.start() }
            }
            .disabled(!det.ready)
        } header: {
            Text("run")
        } footer: {
            Text(det.strategy.note + "  Strategy is locked while running so results stay attributable.")
        }
    }

    // MARK: - anomalies

    private var sectionAnomalies: some View {
        Section("anomalies") {
            LabeledContent("write did not land", value: "\(det.counts.vacated)")
            LabeledContent("stale pattern", value: "\(det.counts.stale)")
            LabeledContent("unexplained", value: "\(det.counts.unexplained)")
        }
    }

    // MARK: - premise

    private var sectionPremise: some View {
        Section {
            LabeledContent("GPU VA reuses", value: "\(det.probe.vaReuses)")
            LabeledContent("VA range", value: String(format: "%.1f MiB", det.probe.rangeMiB))
            LabeledContent("allocations", value: "\(det.probe.allocations)")
            LabeledContent("arena held", value: "\(det.probe.arenaMiB) MiB")
            LabeledContent("GPU bytes", value: String(format: "%.0f MiB",
                                                      Double(det.probe.gpuBytes) / 1_048_576))
        } header: {
            Text("premise probe")
        } footer: {
            Text("If VA range stays flat while allocations climbs, AGX is handing out addresses from a "
               + "pre-mapped region and no per-buffer stage-2 edit happens. If VA reuses is 0, the "
               + "detector could not have fired regardless.")
        }
    }

    // MARK: - recent

    private var sectionRecent: some View {
        Section("recent") {
            if det.recent.isEmpty {
                Text("—")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            } else {
                ForEach(Array(det.recent.enumerated()), id: \.offset) { _, line in
                    Text(line).font(.caption.monospaced())
                }
            }
        }
    }

    // MARK: - caveat

    private var sectionCaveat: some View {
        Section {
            Text("""
            Whether Metal buffer map/unmap reaches the SPTM UAT mutation path is not observable from \
            userland. If AGX sub-allocates inside a pre-mapped heap, no per-buffer stage-2 edit \
            happens and this detector can never fire.

            Only kernel instrumentation or an SRD settles it.
            """)
            .font(.footnote)
        } header: {
            Text("the one thing this cannot settle")
        }
    }

    // MARK: - export

    private var sectionExport: some View {
        Section {
            ShareLink(item: report) {
                Text("share diagnostic report")
            }
        } footer: {
            Text("Paste this into a bug report; it carries everything needed to reproduce the state.")
        }
    }

    private var report: String {
        """
        SPTM race detector v4
        build:      \(det.verdict.text)
        strategy:   \(det.strategy.rawValue)
        phase:      \(det.phase)
        headline:   \(det.headline)
        anomalies:  vacated=\(det.counts.vacated) stale=\(det.counts.stale) unexplained=\(det.counts.unexplained)
        vaReuses:   \(det.probe.vaReuses)
        vaRangeMiB: \(String(format: "%.1f", det.probe.rangeMiB))
        allocations:\(det.probe.allocations)
        arenaMiB:   \(det.probe.arenaMiB)
        gpuMiB:     \(String(format: "%.0f", Double(det.probe.gpuBytes) / 1_048_576))
        vaProbe:    \(det.vaProbeAvailable ? "available" : "unavailable (iOS <18)")
        selfTest:   \(det.selfTest.joined(separator: " | "))
        """
    }
}

@main
struct RacePoCApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}
