import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { auth } from '../firebase';
import {
  User,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  onAuthStateChanged,
  signOut,
  GoogleAuthProvider,
  signInWithPopup,
  onIdTokenChanged
} from 'firebase/auth';
import { doc, getDoc } from 'firebase/firestore';

// Declare chrome types for TypeScript
declare global {
  interface Window {
    chrome?: any;
  }
}

interface AuthUser {
  id: string;
  email: string;
  name?: string;
  isAdmin: boolean;
}

interface AuthContextType {
  user: AuthUser | null;
  loading: boolean;
  signUp: (email: string, password: string, name?: string) => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signInWithGoogle: () => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  // Helper function to send token to extension
  const sendTokenToExtension = async (token: string | null) => {
    try {
      // Get the extension ID from localStorage if we've saved it
      const savedExtensionId = localStorage.getItem('truthguard_extension_id');
      
      // Check if Chrome runtime APIs are available
      if (typeof window.chrome === 'undefined' || !window.chrome.runtime || !window.chrome.runtime.sendMessage) {
        console.log('Chrome extension APIs not available');
        return;
      }

      // Try common extension ID patterns or saved ID
      const extensionIds = savedExtensionId ? [savedExtensionId] : ['kmejojlkkncgfackkipfggbgkmcogmch'];
      
      if (!token) {
        // Send logout message
        for (const id of extensionIds) {
          try {
            window.chrome.runtime.sendMessage(id, { type: 'FIREBASE_LOGOUT' }, (response: any) => {
              if (response && response.success) {
                console.log('✅ Logout message sent to extension');
              }
            });
          } catch (e) {
            // Extension not found, ignore
          }
        }
        return;
      }

      // Send token to extension
      for (const id of extensionIds) {
        try {
          window.chrome.runtime.sendMessage(id, { type: 'FIREBASE_AUTH_TOKEN', token }, (response: any) => {
            if (response && response.success) {
              console.log('✅ Token sent to extension');
            }
          });
        } catch (e) {
          console.log('Extension not found or not responding');
        }
      }
    } catch (e) {
      // Not in a Chrome extension context or extension not installed
      console.log('Could not send message to extension');
    }
  };

  useEffect(() => {
    // Listen for auth changes
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      if (firebaseUser) {
        try {
          // Fetch user data from Firestore to get admin status
          const { db } = await import('../firebase');
          const userDoc = await getDoc(doc(db, 'users', firebaseUser.uid));
          const userData = userDoc.data();

          const authUser: AuthUser = {
            id: firebaseUser.uid,
            email: firebaseUser.email || '',
            name: firebaseUser.displayName || undefined,
            isAdmin: userData?.isAdmin || false
          };
          setUser(authUser);
        } catch (error) {
          console.error('Error fetching user data:', error);
          // Fallback to basic user data if Firestore fetch fails
          const authUser: AuthUser = {
            id: firebaseUser.uid,
            email: firebaseUser.email || '',
            name: firebaseUser.displayName || undefined,
            isAdmin: false
          };
          setUser(authUser);
        }
      } else {
        setUser(null);
      }
      setLoading(false);
    });

    // Also listen for ID token changes to send to extension
    const unsubscribeToken = onIdTokenChanged(auth, async (firebaseUser) => {
      if (firebaseUser) {
        const token = await firebaseUser.getIdToken();
        await sendTokenToExtension(token);
      } else {
        await sendTokenToExtension(null);
      }
    });

    return () => {
      unsubscribe();
      unsubscribeToken();
    };
  }, []);

  const signUp = async (email: string, password: string, name?: string) => {
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);

      // Store additional user data in Firestore
      const { db } = await import('../firebase');
      const { doc, setDoc } = await import('firebase/firestore');

      await setDoc(doc(db, 'users', userCredential.user.uid), {
        identifier: email,
        tier: 'free',
        role: 'user',
        email: email,
        name: name || '',
        isAdmin: false,
        createdAt: new Date(),
        analysesCount: 0,
        lastActive: new Date(),
      });

    } catch (error: any) {
      console.error('Signup error:', error);
      throw new Error(error.message || 'Failed to create account');
    }
  };

  const signIn = async (email: string, password: string) => {
    try {
      await signInWithEmailAndPassword(auth, email, password);
    } catch (error: any) {
      console.error('Sign in error:', error);
      throw new Error(error.message || 'Failed to sign in');
    }
  };

  const signInWithGoogle = async () => {
    try {
      const provider = new GoogleAuthProvider();
      const userCredential = await signInWithPopup(auth, provider);

      // Check if user document exists, if not create it
      const { db } = await import('../firebase');
      const { doc, setDoc, getDoc } = await import('firebase/firestore');

      const userDocRef = doc(db, 'users', userCredential.user.uid);
      const userDoc = await getDoc(userDocRef);

      if (!userDoc.exists()) {
        await setDoc(userDocRef, {
          identifier: userCredential.user.email || '',
          tier: 'free',
          role: 'user',
          //email: userCredential.user.email || '',
          name: userCredential.user.displayName || '',
          isAdmin: false,
          createdAt: new Date(),
          analysesCount: 0,
          lastActive: new Date(),
        });
      }
    } catch (error: any) {
      console.error('Google sign in error:', error);
      throw new Error(error.message || 'Failed to sign in with Google');
    }
  };

  const logout = async () => {
    try {
      await signOut(auth);
    } catch (error: any) {
      console.error('Sign out error:', error);
      throw new Error(error.message || 'Failed to sign out');
    }
  };

  const value = {
    user,
    loading,
    signUp,
    signIn,
    signInWithGoogle,
    logout
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}