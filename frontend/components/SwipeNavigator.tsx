"use client";

import { useEffect, useRef } from "react";
import { usePathname, useRouter } from "next/navigation";

import { SWIPE_ROUTE_ORDER } from "../lib/navigation";

const SWIPE_THRESHOLD_PX = 72;
const SWIPE_MAX_DURATION_MS = 600;
const SWIPE_AXIS_RATIO = 1.25;
const SWIPE_IGNORE_SELECTOR =
  "input, select, textarea, button, a, summary, label, [role='button'], [data-no-swipe], [data-swipe-lock]";

type SwipeState = {
  active: boolean;
  ignore: boolean;
  startX: number;
  startY: number;
  startedAt: number;
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
    startedAt: 0
  });

  useEffect(() => {
    const onTouchStart = (event: TouchEvent): void => {
      const touch = event.touches[0];
      if (!touch || event.touches.length !== 1) {
        swipeRef.current.active = false;
        swipeRef.current.ignore = false;
        return;
      }

      swipeRef.current = {
        active: true,
        ignore: shouldIgnoreSwipeTarget(event.target),
        startX: touch.clientX,
        startY: touch.clientY,
        startedAt: Date.now()
      };
    };

    const onTouchEnd = (event: TouchEvent): void => {
      const swipe = swipeRef.current;
      swipeRef.current.active = false;

      if (!swipe.active || swipe.ignore) {
        return;
      }

      const routeIndex = currentRouteIndex(pathname);
      if (routeIndex === -1) {
        return;
      }

      const touch = event.changedTouches[0];
      if (!touch) {
        return;
      }

      const dx = touch.clientX - swipe.startX;
      const dy = touch.clientY - swipe.startY;
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

      router.push(SWIPE_ROUTE_ORDER[targetIndex]);
    };

    const onTouchCancel = (): void => {
      swipeRef.current.active = false;
      swipeRef.current.ignore = false;
    };

    window.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("touchend", onTouchEnd, { passive: true });
    window.addEventListener("touchcancel", onTouchCancel, { passive: true });

    return () => {
      window.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchend", onTouchEnd);
      window.removeEventListener("touchcancel", onTouchCancel);
    };
  }, [pathname, router]);

  return null;
}
