// Tests for useRequireAuth — the require-auth guard hook.
//
// It reads `user` from localStorage and, via a mount effect, redirects
// to /login when there's no user. It returns the parsed user object (or
// null). We spy on react-router's useNavigate and drive localStorage to
// cover: unauthenticated → redirect, authenticated → pass-through (no
// redirect), and the corrupted-JSON edge case.

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook } from '@testing-library/react';

const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return { ...actual, useNavigate: () => mockNavigate };
});

import useRequireAuth from '../../src/hooks/useRequireAuth';

const setUser = (user) => sessionStorage.setItem('user', JSON.stringify(user));

beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
});

describe('useRequireAuth', () => {
    it('redirects to /login when no user is stored', () => {
        const { result } = renderHook(() => useRequireAuth());
        expect(mockNavigate).toHaveBeenCalledWith('/login');
        // Returns null when unauthenticated.
        expect(result.current).toBeNull();
    });

    it('passes through (no redirect) and returns the user when authenticated', () => {
        const user = { user_id: 7, email: 'a@b.c' };
        setUser(user);
        const { result } = renderHook(() => useRequireAuth());
        expect(mockNavigate).not.toHaveBeenCalled();
        expect(result.current).toEqual(user);
    });

    it('redirects when the stored user is explicitly null', () => {
        sessionStorage.setItem('user', 'null');
        const { result } = renderHook(() => useRequireAuth());
        expect(mockNavigate).toHaveBeenCalledWith('/login');
        expect(result.current).toBeNull();
    });

    it('does not re-redirect on re-render while still authenticated', () => {
        setUser({ user_id: 1 });
        const { rerender } = renderHook(() => useRequireAuth());
        rerender();
        expect(mockNavigate).not.toHaveBeenCalled();
    });

    it('throws on corrupted JSON (JSON.parse is not guarded)', () => {
        sessionStorage.setItem('user', '{not-json');
        // Documents current behavior: a malformed blob bubbles up.
        expect(() => renderHook(() => useRequireAuth())).toThrow();
    });
});
