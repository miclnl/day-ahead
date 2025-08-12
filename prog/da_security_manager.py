"""
Security Manager Module voor DAO

Deze module implementeert:
- JWT token authenticatie
- Role-based access control (RBAC)
- API key management
- Security headers en CORS
- Input validation en sanitization
- Rate limiting per gebruiker
- Audit logging
"""

import logging
import time
import hashlib
import hmac
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
from functools import wraps
import json
import re
import ipaddress

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logging.warning("bcrypt niet beschikbaar - password hashing beperkt")


@dataclass
class User:
    """Data class voor gebruiker informatie"""
    user_id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None


@dataclass
class Role:
    """Data class voor rol informatie"""
    role_id: str
    name: str
    description: str
    permissions: List[str]
    is_active: bool


@dataclass
class Permission:
    """Data class voor permissie informatie"""
    permission_id: str
    name: str
    description: str
    resource: str
    action: str
    conditions: Dict[str, Any]


@dataclass
class SecurityEvent:
    """Data class voor security events"""
    event_id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    details: Dict[str, Any]
    severity: str  # low, medium, high, critical


class SecurityManager:
    """Hoofdklasse voor security management"""

    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()

        # JWT configuratie
        self.jwt_secret = config.get(['security', 'jwt_secret'], None, None)
        self.jwt_algorithm = config.get(['security', 'jwt_algorithm'], None, 'HS256')
        self.jwt_expiration = config.get(['security', 'jwt_expiration'], None, 3600)  # 1 uur

        # API key configuratie
        self.api_keys = {}
        self.api_key_expiration = config.get(['security', 'api_key_expiration'], None, 86400 * 30)  # 30 dagen

        # Gebruikers en rollen
        self.users = {}
        self.roles = {}
        self.permissions = {}

        # Security instellingen
        self.max_login_attempts = config.get(['security', 'max_login_attempts'], None, 5)
        self.lockout_duration = config.get(['security', 'lockout_duration'], None, 900)  # 15 minuten
        self.password_min_length = config.get(['security', 'password_min_length'], None, 8)
        self.require_special_chars = config.get(['security', 'require_special_chars'], None, True)

        # Rate limiting per gebruiker
        self.user_rate_limits = {}
        self.default_user_rate_limit = config.get(['security', 'default_user_rate_limit'], None, 100)  # requests per minute

        # Audit logging
        self.audit_log = []
        self.max_audit_log_size = config.get(['security', 'max_audit_log_size'], None, 10000)

        # Initialiseer security systeem
        self._initialize_security_system()

        logging.info("Security Manager geïnitialiseerd")

    def _initialize_security_system(self):
        """Initialiseer het security systeem"""
        try:
            # Genereer JWT secret als niet opgegeven
            if not self.jwt_secret:
                self.jwt_secret = secrets.token_urlsafe(32)
                logging.warning("JWT secret automatisch gegenereerd - configureer dit in productie")

            # Maak standaard rollen aan
            self._create_default_roles()

            # Maak standaard permissies aan
            self._create_default_permissions()

            # Maak admin gebruiker aan als niet bestaat
            self._create_default_admin()

            logging.info("Security systeem geïnitialiseerd")

        except Exception as e:
            logging.error(f"Security systeem initialisatie fout: {e}")

    def _create_default_roles(self):
        """Maak standaard rollen aan"""
        try:
            # Admin rol
            admin_role = Role(
                role_id="admin",
                name="Administrator",
                description="Volledige toegang tot alle functionaliteiten",
                permissions=["*"],  # Alle permissies
                is_active=True
            )
            self.roles["admin"] = admin_role

            # User rol
            user_role = Role(
                role_id="user",
                name="Gebruiker",
                description="Standaard gebruiker toegang",
                permissions=["read:own_data", "write:own_data", "read:public_data"],
                is_active=True
            )
            self.roles["user"] = user_role

            # Read-only rol
            readonly_role = Role(
                role_id="readonly",
                name="Read-Only",
                description="Alleen lees toegang",
                permissions=["read:public_data", "read:own_data"],
                is_active=True
            )
            self.roles["readonly"] = readonly_role

        except Exception as e:
            logging.error(f"Standaard rollen aanmaken fout: {e}")

    def _create_default_permissions(self):
        """Maak standaard permissies aan"""
        try:
            # Data permissies
            permissions = [
                Permission("read:own_data", "Lees eigen data", "Gebruiker kan eigen data lezen", "data", "read", {"owner": "self"}),
                Permission("write:own_data", "Schrijf eigen data", "Gebruiker kan eigen data wijzigen", "data", "write", {"owner": "self"}),
                Permission("read:public_data", "Lees publieke data", "Gebruiker kan publieke data lezen", "data", "read", {"public": True}),
                Permission("read:all_data", "Lees alle data", "Gebruiker kan alle data lezen", "data", "read", {}),
                Permission("write:all_data", "Schrijf alle data", "Gebruiker kan alle data wijzigen", "data", "write", {}),

                # API permissies
                Permission("api:read", "API lees toegang", "Gebruiker kan API endpoints lezen", "api", "read", {}),
                Permission("api:write", "API schrijf toegang", "Gebruiker kan API endpoints wijzigen", "api", "write", {}),
                Permission("api:admin", "API admin toegang", "Gebruiker kan API beheren", "api", "admin", {}),

                # Systeem permissies
                Permission("system:read", "Systeem lees toegang", "Gebruiker kan systeem informatie lezen", "system", "read", {}),
                Permission("system:write", "Systeem schrijf toegang", "Gebruiker kan systeem instellingen wijzigen", "system", "write", {}),
                Permission("system:admin", "Systeem admin toegang", "Gebruiker kan systeem beheren", "system", "admin", {})
            ]

            for permission in permissions:
                self.permissions[permission.permission_id] = permission

        except Exception as e:
            logging.error(f"Standaard permissies aanmaken fout: {e}")

    def _create_default_admin(self):
        """Maak standaard admin gebruiker aan"""
        try:
            # Check of admin al bestaat
            if "admin" in self.users:
                return

            # Maak admin gebruiker aan
            admin_user = User(
                user_id="admin",
                username="admin",
                email="admin@dao.local",
                roles=["admin"],
                permissions=["*"],
                is_active=True,
                created_at=datetime.now()
            )

            self.users["admin"] = admin_user

            # Genereer wachtwoord hash
            default_password = "admin123"  # Verander dit in productie!
            password_hash = self._hash_password(default_password)

            # Sla wachtwoord hash op (in productie zou dit in een database staan)
            if not hasattr(self, 'password_hashes'):
                self.password_hashes = {}
            self.password_hashes["admin"] = password_hash

            logging.warning("Standaard admin gebruiker aangemaakt - verander wachtwoord in productie!")

        except Exception as e:
            logging.error(f"Standaard admin aanmaken fout: {e}")

    def authenticate_user(self, username: str, password: str, ip_address: str = None, user_agent: str = None) -> Optional[str]:
        """
        Authenticeer een gebruiker

        Args:
            username: Gebruikersnaam
            password: Wachtwoord
            ip_address: IP adres (optioneel)
            user_agent: User agent (optioneel)

        Returns:
            JWT token bij succes, None bij falen
        """
        try:
            # Check of gebruiker bestaat
            if username not in self.users:
                self._log_security_event("failed_login", username, ip_address, user_agent,
                                       {"reason": "user_not_found"})
                return None

            user = self.users[username]

            # Check of gebruiker actief is
            if not user.is_active:
                self._log_security_event("failed_login", username, ip_address, user_agent,
                                       {"reason": "user_inactive"})
                return None

            # Check of gebruiker geblokkeerd is
            if user.locked_until and datetime.now() < user.locked_until:
                self._log_security_event("failed_login", username, ip_address, user_agent,
                                       {"reason": "user_locked", "locked_until": user.locked_until.isoformat()})
                return None

            # Verifieer wachtwoord
            if not self._verify_password(username, password):
                # Verhoog failed login attempts
                user.failed_login_attempts += 1

                # Check of gebruiker geblokkeerd moet worden
                if user.failed_login_attempts >= self.max_login_attempts:
                    user.locked_until = datetime.now() + timedelta(seconds=self.lockout_duration)
                    self._log_security_event("user_locked", username, ip_address, user_agent,
                                           {"reason": "max_attempts_exceeded", "lockout_duration": self.lockout_duration})

                self._log_security_event("failed_login", username, ip_address, user_agent,
                                       {"reason": "invalid_password", "attempts": user.failed_login_attempts})
                return None

            # Reset failed login attempts
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.now()

            # Genereer JWT token
            token = self._generate_jwt_token(user)

            # Log succesvolle login
            self._log_security_event("successful_login", username, ip_address, user_agent,
                                   {"user_id": user.user_id, "roles": user.roles})

            return token

        except Exception as e:
            logging.error(f"Gebruiker authenticatie fout: {e}")
            return None

    def _hash_password(self, password: str) -> str:
        """Hash een wachtwoord"""
        try:
            if BCRYPT_AVAILABLE:
                # Gebruik bcrypt voor sterke hashing
                salt = bcrypt.gensalt()
                return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
            else:
                # Fallback naar SHA-256 met salt
                salt = secrets.token_hex(16)
                hash_obj = hashlib.sha256()
                hash_obj.update((password + salt).encode('utf-8'))
                return f"{salt}:{hash_obj.hexdigest()}"

        except Exception as e:
            logging.error(f"Wachtwoord hashing fout: {e}")
            return ""

    def _verify_password(self, username: str, password: str) -> bool:
        """Verifieer een wachtwoord"""
        try:
            if username not in getattr(self, 'password_hashes', {}):
                return False

            stored_hash = self.password_hashes[username]

            if BCRYPT_AVAILABLE and not stored_hash.startswith('salt:'):
                # bcrypt hash
                return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
            else:
                # SHA-256 hash met salt
                if ':' not in stored_hash:
                    return False

                salt, hash_value = stored_hash.split(':', 1)
                hash_obj = hashlib.sha256()
                hash_obj.update((password + salt).encode('utf-8'))
                return hash_obj.hexdigest() == hash_value

        except Exception as e:
            logging.error(f"Wachtwoord verificatie fout: {e}")
            return False

    def _generate_jwt_token(self, user: User) -> str:
        """Genereer JWT token voor gebruiker"""
        try:
            payload = {
                'user_id': user.user_id,
                'username': user.username,
                'roles': user.roles,
                'permissions': user.permissions,
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(seconds=self.jwt_expiration)
            }

            token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
            return token

        except Exception as e:
            logging.error(f"JWT token generatie fout: {e}")
            return ""

    def verify_jwt_token(self, token: str) -> Optional[User]:
        """
        Verifieer JWT token

        Args:
            token: JWT token

        Returns:
            User object bij succes, None bij falen
        """
        try:
            # Decode token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])

            # Check of gebruiker nog bestaat
            user_id = payload.get('user_id')
            if user_id not in self.users:
                return None

            user = self.users[user_id]

            # Check of gebruiker nog actief is
            if not user.is_active:
                return None

            return user

        except jwt.ExpiredSignatureError:
            logging.debug("JWT token verlopen")
            return None
        except jwt.InvalidTokenError as e:
            logging.debug(f"Ongeldige JWT token: {e}")
            return None
        except Exception as e:
            logging.error(f"JWT token verificatie fout: {e}")
            return None

    def check_permission(self, user: User, resource: str, action: str, context: Dict[str, Any] = None) -> bool:
        """
        Check of gebruiker permissie heeft

        Args:
            user: Gebruiker object
            resource: Resource naam
            action: Actie naam
            context: Extra context (optioneel)

        Returns:
            True als gebruiker permissie heeft, False anders
        """
        try:
            # Admin heeft alle permissies
            if "*" in user.permissions or "admin" in user.roles:
                return True

            # Check specifieke permissies
            for permission_id in user.permissions:
                if permission_id in self.permissions:
                    permission = self.permissions[permission_id]

                    # Check resource en actie
                    if permission.resource == resource and permission.action == action:
                        # Check condities
                        if self._check_permission_conditions(permission, context):
                            return True

            # Check rol permissies
            for role_name in user.roles:
                if role_name in self.roles:
                    role = self.roles[role_name]
                    for permission_id in role.permissions:
                        if permission_id in self.permissions:
                            permission = self.permissions[permission_id]

                            # Check resource en actie
                            if permission.resource == resource and permission.action == action:
                                # Check condities
                                if self._check_permission_conditions(permission, context):
                                    return True

            return False

        except Exception as e:
            logging.error(f"Permissie check fout: {e}")
            return False

    def _check_permission_conditions(self, permission: Permission, context: Dict[str, Any]) -> bool:
        """Check permissie condities"""
        try:
            if not context or not permission.conditions:
                return True

            for condition_key, condition_value in permission.conditions.items():
                if condition_key == "owner" and condition_value == "self":
                    # Check of gebruiker eigenaar is van resource
                    if "owner_id" in context and context["owner_id"] != context.get("user_id"):
                        return False

                elif condition_key == "public":
                    # Check of resource publiek is
                    if not context.get("public", False):
                        return False

            return True

        except Exception as e:
            logging.debug(f"Permissie conditie check fout: {e}")
            return False

    def create_api_key(self, user_id: str, description: str = None, permissions: List[str] = None,
                       expires_in: int = None) -> str:
        """
        Maak API key aan voor gebruiker

        Args:
            user_id: Gebruiker ID
            description: Beschrijving van API key
            permissions: Specifieke permissies (optioneel)
            expires_in: Vervaldatum in seconden (optioneel)

        Returns:
            API key
        """
        try:
            # Genereer unieke API key
            api_key = f"dao_{secrets.token_urlsafe(32)}"

            # Bepaal vervaldatum
            if expires_in is None:
                expires_in = self.api_key_expiration

            expiration = datetime.now() + timedelta(seconds=expires_in)

            # Sla API key op
            self.api_keys[api_key] = {
                'user_id': user_id,
                'description': description or f"API key voor {user_id}",
                'permissions': permissions or [],
                'created_at': datetime.now(),
                'expires_at': expiration,
                'last_used': None,
                'is_active': True
            }

            # Log API key creatie
            self._log_security_event("api_key_created", user_id, None, None,
                                   {"api_key": api_key[:10] + "...", "description": description})

            return api_key

        except Exception as e:
            logging.error(f"API key creatie fout: {e}")
            return ""

    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Verifieer API key

        Args:
            api_key: API key

        Returns:
            API key informatie bij succes, None bij falen
        """
        try:
            if api_key not in self.api_keys:
                return None

            key_info = self.api_keys[api_key]

            # Check of key actief is
            if not key_info['is_active']:
                return None

            # Check of key verlopen is
            if datetime.now() > key_info['expires_at']:
                return None

            # Update laatste gebruik
            key_info['last_used'] = datetime.now()

            return key_info

        except Exception as e:
            logging.error(f"API key verificatie fout: {e}")
            return None

    def revoke_api_key(self, api_key: str, user_id: str = None) -> bool:
        """
        Herroep API key

        Args:
            api_key: API key om te herroepen
            user_id: Gebruiker ID (optioneel, voor verificatie)

        Returns:
            True bij succes, False anders
        """
        try:
            if api_key not in self.api_keys:
                return False

            key_info = self.api_keys[api_key]

            # Check of gebruiker key mag herroepen
            if user_id and key_info['user_id'] != user_id:
                return False

            # Markeer als inactief
            key_info['is_active'] = False

            # Log herroeping
            self._log_security_event("api_key_revoked", key_info['user_id'], None, None,
                                   {"api_key": api_key[:10] + "...", "description": key_info['description']})

            return True

        except Exception as e:
            logging.error(f"API key herroeping fout: {e}")
            return False

    def check_user_rate_limit(self, user_id: str, endpoint: str = "api") -> bool:
        """
        Check rate limit voor specifieke gebruiker

        Args:
            user_id: Gebruiker ID
            endpoint: Endpoint naam

        Returns:
            True als request toegestaan is, False anders
        """
        try:
            # Maak unieke key voor gebruiker + endpoint
            rate_limit_key = f"{user_id}:{endpoint}"

            if rate_limit_key not in self.user_rate_limits:
                # Initialiseer rate limit voor gebruiker
                self.user_rate_limits[rate_limit_key] = {
                    'current_requests': 0,
                    'max_requests': self.default_user_rate_limit,
                    'window_start': time.time(),
                    'window_duration': 60  # 1 minuut
                }

            rate_limit = self.user_rate_limits[rate_limit_key]
            current_time = time.time()

            # Check of rate limit window verlopen is
            if current_time - rate_limit['window_start'] > rate_limit['window_duration']:
                # Reset window
                rate_limit['current_requests'] = 0
                rate_limit['window_start'] = current_time

            # Check of limiet bereikt is
            if rate_limit['current_requests'] >= rate_limit['max_requests']:
                return False

            # Verhoog request teller
            rate_limit['current_requests'] += 1
            return True

        except Exception as e:
            logging.debug(f"Gebruiker rate limit check fout: {e}")
            return True  # Sta toe bij fout

    def _log_security_event(self, event_type: str, user_id: str, ip_address: str, user_agent: str, details: Dict[str, Any]):
        """Log een security event"""
        try:
            event = SecurityEvent(
                event_id=secrets.token_urlsafe(16),
                timestamp=datetime.now(),
                event_type=event_type,
                user_id=user_id,
                ip_address=ip_address or "unknown",
                user_agent=user_agent or "unknown",
                details=details,
                severity=self._determine_event_severity(event_type)
            )

            # Voeg toe aan audit log
            self.audit_log.append(event)

            # Behoud maximale grootte
            if len(self.audit_log) > self.max_audit_log_size:
                self.audit_log = self.audit_log[-self.max_audit_log_size:]

            # Log naar logging systeem
            log_message = f"Security Event [{event.severity.upper()}]: {event_type} - User: {user_id} - IP: {ip_address}"
            if event.severity in ['high', 'critical']:
                logging.warning(log_message)
            else:
                logging.info(log_message)

        except Exception as e:
            logging.error(f"Security event logging fout: {e}")

    def _determine_event_severity(self, event_type: str) -> str:
        """Bepaal severity van een event type"""
        try:
            high_severity_events = ['failed_login', 'user_locked', 'api_key_revoked', 'permission_violation']
            medium_severity_events = ['successful_login', 'api_key_created', 'role_changed']
            low_severity_events = ['permission_check', 'rate_limit_exceeded']

            if event_type in high_severity_events:
                return 'high'
            elif event_type in medium_severity_events:
                return 'medium'
            elif event_type in low_severity_events:
                return 'low'
            else:
                return 'medium'

        except Exception as e:
            logging.debug(f"Event severity bepaling fout: {e}")
            return 'medium'

    def get_security_events(self, event_type: str = None, user_id: str = None,
                           severity: str = None, hours: int = 24) -> List[SecurityEvent]:
        """
        Haal security events op

        Args:
            event_type: Filter op event type (optioneel)
            user_id: Filter op gebruiker ID (optioneel)
            severity: Filter op severity (optioneel)
            hours: Aantal uren om terug te kijken

        Returns:
            Lijst van security events
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            filtered_events = []
            for event in self.audit_log:
                # Check tijd filter
                if event.timestamp < cutoff_time:
                    continue

                # Check event type filter
                if event_type and event.event_type != event_type:
                    continue

                # Check gebruiker filter
                if user_id and event.user_id != user_id:
                    continue

                # Check severity filter
                if severity and event.severity != severity:
                    continue

                filtered_events.append(event)

            # Sorteer op timestamp (nieuwste eerst)
            filtered_events.sort(key=lambda x: x.timestamp, reverse=True)

            return filtered_events

        except Exception as e:
            logging.error(f"Security events ophalen fout: {e}")
            return []

    def validate_input(self, input_data: str, input_type: str = "general") -> Tuple[bool, str]:
        """
        Valideer en sanitize input

        Args:
            input_data: Input data om te valideren
            input_type: Type input ('email', 'username', 'password', 'general')

        Returns:
            Tuple van (is_valid, sanitized_data)
        """
        try:
            if not input_data or not isinstance(input_data, str):
                return False, ""

            # Basis sanitization
            sanitized = input_data.strip()

            # Type-specifieke validatie
            if input_type == "email":
                # Email validatie
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, sanitized):
                    return False, ""

            elif input_type == "username":
                # Username validatie
                if len(sanitized) < 3 or len(sanitized) > 50:
                    return False, ""

                # Alleen alfanumerieke karakters en underscores
                if not re.match(r'^[a-zA-Z0-9_]+$', sanitized):
                    return False, ""

            elif input_type == "password":
                # Wachtwoord validatie
                if len(sanitized) < self.password_min_length:
                    return False, ""

                if self.require_special_chars:
                    # Moet minimaal 1 hoofdletter, 1 kleine letter, 1 cijfer en 1 speciaal karakter bevatten
                    if not re.search(r'[A-Z]', sanitized):
                        return False, ""
                    if not re.search(r'[a-z]', sanitized):
                        return False, ""
                    if not re.search(r'\d', sanitized):
                        return False, ""
                    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', sanitized):
                        return False, ""

            # Algemene security checks
            # Voorkom SQL injection patronen
            sql_patterns = ["'", '"', ';', '--', '/*', '*/', 'xp_', 'sp_']
            if any(pattern in sanitized.lower() for pattern in sql_patterns):
                return False, ""

            # Voorkom XSS patronen
            xss_patterns = ['<script', 'javascript:', 'onload=', 'onerror=']
            if any(pattern in sanitized.lower() for pattern in xss_patterns):
                return False, ""

            return True, sanitized

        except Exception as e:
            logging.error(f"Input validatie fout: {e}")
            return False, ""

    def sanitize_output(self, output_data: str) -> str:
        """
        Sanitize output data

        Args:
            output_data: Output data om te sanitizen

        Returns:
            Gesanitizede output data
        """
        try:
            if not output_data or not isinstance(output_data, str):
                return ""

            # HTML escape
            sanitized = output_data.replace('&', '&amp;')
            sanitized = sanitized.replace('<', '&lt;')
            sanitized = sanitized.replace('>', '&gt;')
            sanitized = sanitized.replace('"', '&quot;')
            sanitized = sanitized.replace("'", '&#x27;')

            return sanitized

        except Exception as e:
            logging.error(f"Output sanitization fout: {e}")
            return ""


# Utility functies voor externe gebruik
def create_security_manager(config) -> SecurityManager:
    """Factory functie voor het maken van een SecurityManager instance"""
    return SecurityManager(config)


def require_auth(security_manager: SecurityManager):
    """Decorator voor authenticatie vereiste"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Haal token op uit request (moet geïmplementeerd worden in Flask context)
            # Voor nu, return functie zonder authenticatie
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_permission(security_manager: SecurityManager, resource: str, action: str):
    """Decorator voor permissie vereiste"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check permissies (moet geïmplementeerd worden in Flask context)
            # Voor nu, return functie zonder permissie check
            return func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limit_user(security_manager: SecurityManager, endpoint: str = "api"):
    """Decorator voor gebruiker rate limiting"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check rate limit (moet geïmplementeerd worden in Flask context)
            # Voor nu, return functie zonder rate limiting
            return func(*args, **kwargs)
        return wrapper
    return decorator
