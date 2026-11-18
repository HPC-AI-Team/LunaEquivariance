"""eSEN frame convention and retained-row Wigner installation."""

from types import MethodType


from LunaEquivariance.so2 import rotation
from LunaEquivariance.so2.rotation.interface import selected_local_rows


def install_sparse_wigner(model, *, official: bool = False) -> str:
    """Install retained-m Wigner evaluation on one conservative eSEN model."""
    backbone = model.backbone
    lmax, mmax = backbone.lmax, backbone.mmax
    selected_local_rows(lmax, mmax)
    if official and backbone.direct_forces:
        raise ValueError("sparse Wigner requires conservative eSEN forces")

    def configured_branchless_wigner(edge_rot_mat, jd_buffers, requested_lmax):
        if requested_lmax != lmax:
            raise ValueError("unexpected eSEN Wigner lmax")
        return sparse_branchless_wigner(edge_rot_mat, jd_buffers, lmax, mmax)

    backbone.so2_wigner = configured_branchless_wigner
    if official:
        from fairchem.core.models.esen.common.rotation import init_edge_rot_mat

        def configured_rotmat_and_wigner(module, edge_distance_vecs):
            edge_rot_mat = init_edge_rot_mat(edge_distance_vecs, rot_clip=True)
            jd_buffers = [
                getattr(module, f"Jd_{degree}").type(edge_rot_mat.dtype)
                for degree in range(lmax + 1)
            ]
            wigner = configured_branchless_wigner(edge_rot_mat, jd_buffers, module.lmax)
            return edge_rot_mat, wigner, wigner

        backbone.get_rotmat_and_wigner = MethodType(
            configured_rotmat_and_wigner, backbone
        )
        return "official conservative analytic physical-sparse retained-m Wigner"
    return "analytic physical-sparse retained-m Wigner Forward/BW1/BW2"


def sparse_branchless_wigner(edge_rot_mat, jd_buffers, lmax, mmax):
    """eSEN pole semantics with an analytic retained-m Wigner payload."""
    if len(jd_buffers) != lmax + 1:
        raise ValueError("Wigner Jd buffers do not match lmax")
    return rotation(
        edge_rot_mat, jd_buffers, mmax=mmax, convention="pole_clamped"
    )
