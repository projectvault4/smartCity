import { createContext, useContext, useState, useCallback, ReactNode } from 'react';

export type UserRole = 'admin' | 'citizen';

export interface User {
  role: UserRole;
  name: string;
  ward?: string;     // citizen only
  age?: number;      // citizen only
  phone?: string;    // citizen only
  tags?: string[];   // citizen only — derived risk factors
  email: string;
}

interface AuthContextValue {
  user: User | null;
  login: (email: string, password: string, role: UserRole) => boolean;
  register: (user: User) => void;
  logout: () => void;
}

// Demo credentials — swap for real API auth later
const DEMO_CREDENTIALS: Record<UserRole, { email: string; password: string; user: User }> = {
  admin: {
    email: 'admin@foresightx.city',
    password: 'admin123',
    user: { role: 'admin', name: 'City Admin', email: 'admin@foresightx.city' },
  },
  citizen: {
    email: 'citizen@foresightx.city',
    password: 'member123',
    user: {
      role: 'citizen',
      name: 'ForeSightX Member',
      ward: 'Channasandra',
      age: 30,
      phone: '+91 90000 00000',
      email: 'citizen@foresightx.city',
    },
  },
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    try {
      const stored = sessionStorage.getItem('fsx_user');
      return stored ? (JSON.parse(stored) as User) : null;
    } catch {
      return null;
    }
  });

  const login = useCallback((email: string, password: string, role: UserRole): boolean => {
    const cred = DEMO_CREDENTIALS[role];
    if (email.trim() === cred.email && password === cred.password) {
      setUser(cred.user);
      sessionStorage.setItem('fsx_user', JSON.stringify(cred.user));
      return true;
    }
    return false;
  }, []);

  const register = useCallback((newUser: User) => {
    setUser(newUser);
    sessionStorage.setItem('fsx_user', JSON.stringify(newUser));
  }, []);

  const logout = useCallback(() => {
    setUser(null);
    sessionStorage.removeItem('fsx_user');
  }, []);

  return <AuthContext.Provider value={{ user, login, register, logout }}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>');
  return ctx;
}
