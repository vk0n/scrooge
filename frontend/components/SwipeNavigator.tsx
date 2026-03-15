"use client";

import { useEffect, useRef } from "react";
import { usePathname, useRouter } from "next/navigation";

import { SWIPE_ROUTE_ORDER } from "../lib/navigation";

const SWIPE_MAX_DURATION_MS = 900;
const SWIPE_AXIS_RATIO = 1.1;
const SWIPE_THRESHOLD_PX = 48;
const SWIPE_NAV_COOLDOWN_MS = 900;
const SWIPE_IGNORE_SELECTOR =
  "input, select, textarea, button, a, summary, label, [role='button'], [data-no-swipe], [data-swipe-lock]";

type SwipeState = {
  active: boolean;
  ignore: boolean;
  startX: number;
  startY: number;
  lastX: number;
  lastY: number;
  startedAt: number;
  pointerId: number | null;
};

function resolveTarget(target: EventTarget | null): HTMLElement | null {
  return target instanceof HTMLElement ? target : null;
}

function shouldIgnoreSwipeTarget(target: EventTarget | null): boolean {
  const element = resolveTarget(target);
  return Boolean(element?.closest(SWIPE_IGNORE_SELECTOR));
}

function currentRouteIndex(pathname: string): number {
  return SWIPE_ROUTE_ORDER.findIndex((route) => pathname === route || pathname.startsWith(`${route}/`));
}

export default function SwipeNavigator(): null {
  const pathname = usePathname();
  const router = useRouter();
  const swipeRef = useRef<SwipeState>({
    active: false,
    ignore: false,
    startX: 0,
    startY: 0,
    lastX: 0,
    lastY: 0,
    startedAt: 0,
    pointerId: null
  });
  const lastNavigationAtRef = useRef<number>(0);

  useEffect(() => {
    const eventTarget = window;
    const hasPointerEvents =
      typeof (eventTarget as Window & { PointerEvent?: unknown }).PointerEvent === "function";

    const resetSwipe = (): void => {
      swipeRef.current.active = false;
      swipeRef.current.ignore = false;
      swipeRef.current.pointerId = null;
    };

    const beginSwipe = ({
      target,
      x,
      y,
      pointerId
    }: {
      target: EventTarget | null;
      x: number;
      y: number;
      pointerId?: number | null;
    }): void => {
      swipeRef.current = {
        active: true,
        ignore: shouldIgnoreSwipeTarget(target),
        startX: x,
        startY: y,
        lastX: x,
        lastY: y,
        startedAt: Date.now(),
        pointerId: pointerId ?? null
      };
    };

    const updateSwipe = ({
      x,
      y,
      pointerId
    }: {
      x: number;
      y: number;
      pointerId?: number | null;
    }): void => {
      const swipe = swipeRef.current;
      if (!swipe.active || swipe.ignore) {
        return;
      }
      if (swipe.pointerId !== null && pointerId !== null && swipe.pointerId !== pointerId) {
        return;
      }

      swipe.lastX = x;
      swipe.lastY = y;
    };

    const completeSwipe = ({
      x,
      y,
      pointerId
    }: {
      x?: number;
      y?: number;
      pointerId?: number | null;
    }): void => {
      const swipe = swipeRef.current;
      resetSwipe();

      if (!swipe.active || swipe.ignore) {
        return;
      }
      if (swipe.pointerId !== null && pointerId !== null && swipe.pointerId !== pointerId) {
        return;
      }

      const routeIndex = currentRouteIndex(pathname);
      if (routeIndex === -1) {
        return;
      }

      if (Date.now() - lastNavigationAtRef.current < SWIPE_NAV_COOLDOWN_MS) {
        return;
      }

      const endX = x ?? swipe.lastX;
      const endY = y ?? swipe.lastY;
      const dx = endX - swipe.startX;
      const dy = endY - swipe.startY;
      const elapsedMs = Date.now() - swipe.startedAt;
      const absDx = Math.abs(dx);
      const absDy = Math.abs(dy);

      if (elapsedMs > SWIPE_MAX_DURATION_MS || absDx < SWIPE_THRESHOLD_PX || absDx < absDy * SWIPE_AXIS_RATIO) {
        return;
      }

      const targetIndex = dx < 0 ? routeIndex + 1 : routeIndex - 1;
      if (targetIndex < 0 || targetIndex >= SWIPE_ROUTE_ORDER.length) {
        return;
      }

      lastNavigationAtRef.current = Date.now();
      router.push(SWIPE_ROUTE_ORDER[targetIndex]);
    };

    if (hasPointerEvents) {
      const onPointerDown = (event: PointerEvent): void => {
        if (!event.isPrimary || event.pointerType !== "touch") {
          return;
        }
        beginSwipe({
          target: event.target,
          x: event.clientX,
          y: event.clientY,
          pointerId: event.pointerId
        });
      };

      const onPointerMove = (event: PointerEvent): void => {
        if (event.pointerType !== "touch") {
          return;
        }
        updateSwipe({
          x: event.clientX,
          y: event.clientY,
          pointerId: event.pointerId
        });
      };

      const onPointerUp = (event: PointerEvent): void => {
        if (event.pointerType !== "touch") {
          return;
        }
        completeSwipe({
          x: event.clientX,
          y: event.clientY,
          pointerId: event.pointerId
        });
      };

      const onPointerCancel = (): void => {
        resetSwipe();
      };

      eventTarget.addEventListener("pointerdown", onPointerDown, { passive: true });
      eventTarget.addEventListener("pointermove", onPointerMove, { passive: true });
      eventTarget.addEventListener("pointerup", onPointerUp, { passive: true });
      eventTarget.addEventListener("pointercancel", onPointerCancel, { passive: true });

      return () => {
        eventTarget.removeEventListener("pointerdown", onPointerDown);
        eventTarget.removeEventListener("pointermove", onPointerMove);
        eventTarget.removeEventListener("pointerup", onPointerUp);
        eventTarget.removeEventListener("pointercancel", onPointerCancel);
      };
    }

    const onTouchStart = (event: TouchEvent): void => {
      const touch = event.touches[0];
      if (!touch || event.touches.length !== 1) {
        resetSwipe();
        return;
      }
      beginSwipe({
        target: event.target,
        x: touch.clientX,
        y: touch.clientY
      });
    };

    const onTouchMove = (event: TouchEvent): void => {
      const touch = event.touches[0];
      if (!touch || event.touches.length !== 1) {
        resetSwipe();
        return;
      }
      updateSwipe({
        x: touch.clientX,
        y: touch.clientY
      });
    };

    const onTouchEnd = (event: TouchEvent): void => {
      const touch = event.changedTouches[0];
      completeSwipe({
        x: touch?.clientX,
        y: touch?.clientY
      });
    };

    const onTouchCancel = (): void => {
      resetSwipe();
    };

    eventTarget.addEventListener("touchstart", onTouchStart, { passive: true });
    eventTarget.addEventListener("touchmove", onTouchMove, { passive: true });
    eventTarget.addEventListener("touchend", onTouchEnd, { passive: true });
    eventTarget.addEventListener("touchcancel", onTouchCancel, { passive: true });

    return () => {
      eventTarget.removeEventListener("touchstart", onTouchStart);
      eventTarget.removeEventListener("touchmove", onTouchMove);
      eventTarget.removeEventListener("touchend", onTouchEnd);
      eventTarget.removeEventListener("touchcancel", onTouchCancel);
    };
  }, [pathname, router]);

  return null;
}
