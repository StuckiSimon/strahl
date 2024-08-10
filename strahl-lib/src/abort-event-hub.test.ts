import { describe, expect, it, vi } from "vitest";
import { buildAbortEventHub } from "./abort-event-hub";

describe("abort-event-hub", () => {
  describe("buildAbortEventHub", () => {
    it("should call notifiers when triggerAbort is called", () => {
      const mock = vi.fn();

      const { setDestructionNotifier, triggerAbort } = buildAbortEventHub();
      setDestructionNotifier("test", mock);
      triggerAbort();

      expect(mock).toHaveBeenCalledOnce();
    });

    it("should also call notifier if prior abortion was registered", () => {
      const mock = vi.fn();

      const { setDestructionNotifier, triggerAbort } = buildAbortEventHub();
      triggerAbort();
      setDestructionNotifier("test", mock);

      expect(mock).toHaveBeenCalledOnce();
    });

    it("should overwrite notifiers with same id", () => {
      const mock = vi.fn();
      const mockUnused = vi.fn();

      const { setDestructionNotifier, triggerAbort } = buildAbortEventHub();
      setDestructionNotifier("test", mockUnused);
      setDestructionNotifier("test", mock);
      triggerAbort();

      expect(mock).toHaveBeenCalledOnce();
      expect(mockUnused).not.toHaveBeenCalled();
    });
  });
});
